/*
  Gesture Pad firmware — XIAO ESP32S3 Sense
  ------------------------------------------------------------
  Arduino IDE settings (Tools menu):
    Board:            XIAO_ESP32S3
    USB Mode:         USB-OTG (TinyUSB)
    USB CDC On Boot:  Disabled        (this sketch creates its own CDC port)
    PSRAM:            OPI PSRAM       (frame buffers + model arena live in PSRAM)
    Partition Scheme: anything with >= 3 MB app (e.g. "Huge APP" or "8M with spiffs")
  Libraries:
    - ArduinoJson (v7)  by Benoit Blanchon
    - VL53L1X           by Pololu
    - Your Edge Impulse library (Sketch > Include Library > Add .ZIP Library)
  If upload fails after flashing this once: hold BOOT, plug in USB, release, upload.

  What it does:
    - USB composite device: keyboard + consumer (media) + mouse + CDC config port
    - 5 keys + encoder push (2x3 matrix), encoder rotation
    - Rule-based motion gestures: palm cover ("Ben 10 slap"), swipe L->R, swipe R->L
      Fuzzy gate: confidence = min(mu_close(ToF), mu_cover(camera coverage))
    - Hand poses from the Edge Impulse model (every label except "none" becomes a
      mappable input), running on core 1 so swipe tracking keeps its frame rate
    - Every input remappable from the configurator, config saved in NVS
    - Debug stream (160x120 grayscale + telemetry) only while the configurator asks
  LD2410B presence sensing is intentionally not implemented yet.
*/

// ============ MODEL ============
// Set to 0 to build without the Edge Impulse library (rule-based gestures only).
#define USE_ML 1
#if USE_ML
#include <rokatu-project-1_inferencing.h>   // change if you retrain under a new project name
#endif

#include <Arduino.h>
#include "USB.h"
#include "USBHIDKeyboard.h"
#include "USBHIDConsumerControl.h"
#include "USBHIDMouse.h"
#include <Wire.h>
#include <VL53L1X.h>
#include <Preferences.h>
#include <ArduinoJson.h>
#include "esp_camera.h"

#if ARDUINO_USB_CDC_ON_BOOT
#error "Set Tools > USB CDC On Boot to Disabled"
#endif
#if ARDUINO_USB_MODE
#error "Set Tools > USB Mode to USB-OTG (TinyUSB)"
#endif

#define FW_VERSION "0.2.2"

// ============ PINS (from the KiCad schematic) ============
// Diodes: anode on the switch/column side, cathode on the row -> rows are driven LOW
// one at a time and columns are read with pull-ups.
static const uint8_t ROW_PINS[2] = {D3, D8};       // Row0, Row1
static const uint8_t COL_PINS[3] = {D0, D1, D2};   // Col0, Col1, Col2
#define ENC_A_PIN   D9
#define ENC_B_PIN   D10
#define LED_PIN     D6
#define TOF_SDA     D4
#define TOF_SCL     D5
// D7 (GPIO44) reserved for LD2410B TX later.

// Set to 0 if your Pololu VL53L1X library is too old to have setROISize()
#define TOF_HAS_ROI 1

// ============ INPUTS ============
enum InputId : uint8_t {
  IN_K1, IN_K2, IN_K3, IN_K4, IN_K5, IN_ENC_PUSH, IN_ENC_CW, IN_ENC_CCW,
  IN_PALM, IN_SWIPE_LR, IN_SWIPE_RL,
  IN_ML0,              // model poses start here
  IN_FIXED = IN_ML0
};
#define MAX_ML 8
#define IN_MAX (IN_FIXED + MAX_ML)

static char IN_NAMES[IN_MAX][24] = {
  "k1", "k2", "k3", "k4", "k5", "enc_push", "enc_cw", "enc_ccw",
  "palm", "swipe_lr", "swipe_rl"
};
static int inCount = IN_FIXED;           // fixed inputs + model poses
static int mlCount = 0;                  // model poses (labels except "none")
static int mlModelIdx[MAX_ML];           // pose slot -> model label index
static int mlNoneIdx = -1;

// Which input sits at each electrical matrix position [row][col].
// The configurator names keys by where they sit on the board:
//   Key 1  Key 2  [knob]
//   Key 3  Key 4  Key 5
// If a keycap in the app lights up for the wrong physical key, swap entries here.
static const uint8_t MATRIX_MAP[2][3] = {
  {IN_K1, IN_K2, IN_ENC_PUSH},   // Row0: SW1, SW2, encoder push (via D6)
  {IN_K3, IN_K4, IN_K5}          // Row1: SW3, SW4, SW5
};

// ============ ACTIONS ============
enum ActType : uint8_t { T_NONE, T_KEY, T_MEDIA, T_MOUSE, T_WHEEL };
static const char* TYPE_NAMES[] = {"none", "key", "media", "mouse", "wheel"};
enum ActMode : uint8_t { M_TAP, M_HOLD, M_BOTH };  // BOTH = tap on start and on end
static const char* MODE_NAMES[] = {"tap", "hold", "both"};

struct Action {
  uint8_t type = T_NONE;
  uint8_t mods = 0;     // bit0 LCtrl, 1 LShift, 2 LAlt, 3 LWin, 4 RCtrl, 5 RShift, 6 RAlt, 7 RWin
  int32_t code = 0;     // HID keyboard usage / consumer usage / mouse button mask / wheel amount
  uint8_t mode = M_TAP;
};

// ============ TUNING (X-macro keeps JSON + defaults in one place) ============
#define TUNE_FIELDS \
  X(int,   tofFull,      80)   /* mm: at or below -> fully "close" */            \
  X(int,   tofZero,      220)  /* mm: at or above -> not close at all */         \
  X(int,   tofOffset,    0)    /* mm added to every ToF reading */               \
  X(int,   roi,          8)    /* ToF ROI size in SPADs (4..16) */               \
  X(float, covLow,       0.20f)/* camera coverage where mu_cover starts rising */\
  X(float, covHigh,      0.55f)/* coverage where mu_cover = 1 */                 \
  X(float, confOn,       0.60f)/* fuzzy confidence needed to accept a swipe */   \
  X(float, trackCov,     0.05f)/* coverage that starts/ends a motion window */   \
  X(int,   diffThr,      28)   /* per-pixel background difference threshold */   \
  X(float, swipeTravel,  0.35f)/* min horizontal travel, fraction of frame */    \
  X(int,   swipeMaxMs,   800)  /* max swipe duration */                          \
  X(float, palmDark,     0.35f)/* frame mean < this * background -> covered */   \
  X(float, palmExit,     0.55f)/* frame mean > this * background -> uncovered */ \
  X(int,   palmHoldMs,   150)  /* how long it must stay covered to count */      \
  X(int,   cooldownMs,   400)  /* ignore new motion gestures after one fires */  \
  X(int,   mirror,       0)    /* 1 = flip swipe direction */                    \
  X(int,   camRotate,    1)    /* 1 = rotate image 180 deg (camera upside down) */\
  X(int,   lockExposure, 1)    /* 1 = lock AEC/AGC after warm-up */              \
  X(int,   mlOn,         1)    /* run the pose model */                          \
  X(float, mlThr,        0.80f)/* min model score for a frame to count */        \
  X(int,   mlFrames,     3)    /* consecutive confident frames to trigger */     \
  X(int,   mlRelease,    2)    /* frames without the pose before it ends */      \
  X(int,   mlCooldownMs, 1500) /* ignore new poses after one ends */             \
  X(int,   mlMaxDist,    0)    /* mm; 0 = off. Poses farther than this ignored */\
  X(int,   encSteps,     4)    /* encoder transitions per detent (EC11: 4) */    \
  X(int,   encReverse,   0)                                                      \
  X(int,   streamFps,    10)

struct Tune {
#define X(t, n, d) t n = d;
  TUNE_FIELDS
#undef X
};

struct Config {
  Action map[IN_MAX];
  Tune tune;
};

static Config cfg;
static Preferences prefs;

// ============ USB ============
USBHIDKeyboard Keyboard;
USBHIDConsumerControl Consumer;
USBHIDMouse Mouse;
USBCDC Link;

static SemaphoreHandle_t linkMtx;
static volatile bool hostAttached = false;
static volatile bool streamOn = false;

// ============ INPUT EVENT QUEUE (vision/ML tasks -> loop) ============
enum Phase : uint8_t { PH_TAP, PH_DOWN, PH_UP };
struct InputEvt { uint8_t id; uint8_t phase; };
static QueueHandle_t inputQ;

static inline void postInput(uint8_t id, uint8_t phase) {
  InputEvt e{id, phase};
  xQueueSend(inputQ, &e, 0);
}

// ============ SENSORS / VISION STATE ============
static VL53L1X tof;
static bool tofOk = false;
static volatile bool tofReconfig = false;
static volatile bool camReconfig = false;
static volatile bool palmActive = false;
static volatile bool swipeWindow = false;
static volatile uint32_t gWinStart = 0;
static volatile int  gTofDist = 9999;

// Camera diagnostics (reported to the configurator)
static bool camOk = false;
static bool camYuv = false;              // fell back to YUV422 (luma extracted in software)
static int  camErr = 0;                  // esp_err_t of the last failed init
static volatile uint32_t camFrames = 0;  // frames successfully processed
static volatile int camBadW = 0, camBadH = 0, camBadLen = 0;

static const int FW = 160, FH = 120, FN = FW * FH;   // camera frame (QQVGA), streamed as-is
static const int SW = 80, SH = 60, SN = SW * SH;     // rule-based processing resolution

// ============ MODEL STATE ============
#if USE_ML
static uint8_t* mlFrame = nullptr;                   // 160x120 copy handed to the ML task
static uint8_t  mlIn[EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT];
static SemaphoreHandle_t mlSem;
static volatile bool mlFrameReady = false;
static volatile float mlScores[EI_CLASSIFIER_LABEL_COUNT];
static volatile int mlMs = 0;
static volatile int mlActiveSlot = -1;
#endif

// ============ PACKET OUTPUT ============
// Packet: A5 5A | type u8 | len u32 LE | payload
// type 1 = frame (u16 w, u16 h, w*h gray bytes), type 2 = JSON text
static void sendPacket(uint8_t type, const uint8_t* data, uint32_t len) {
  if (!hostAttached) return;   // don't gate on Link's "connected" flag: it needs DTR *and* RTS
  uint8_t h[7] = {0xA5, 0x5A, type, (uint8_t)len, (uint8_t)(len >> 8),
                  (uint8_t)(len >> 16), (uint8_t)(len >> 24)};
  xSemaphoreTake(linkMtx, portMAX_DELAY);
  Link.write(h, 7);
  Link.write(data, len);
  xSemaphoreGive(linkMtx);
}

static void sendJson(JsonDocument& d) {
  String s;
  serializeJson(d, s);
  sendPacket(2, (const uint8_t*)s.c_str(), s.length());
}

// ============ MODEL LABELS -> INPUTS ============
static void setupModelInputs() {
#if USE_ML
  for (int i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
    const char* lbl = ei_classifier_inferencing_categories[i];
    if (strcmp(lbl, "none") == 0 || strcmp(lbl, "background") == 0 || strcmp(lbl, "idle") == 0) {
      mlNoneIdx = i;
      continue;
    }
    if (mlCount >= MAX_ML) break;
    mlModelIdx[mlCount] = i;
    snprintf(IN_NAMES[IN_ML0 + mlCount], sizeof(IN_NAMES[0]), "ml_%s", lbl);
    mlCount++;
  }
  inCount = IN_FIXED + mlCount;
#endif
}

// ============ CONFIG <-> JSON ============
static void setDefaults() {
  cfg = Config();
  auto key = [](uint8_t mods, int32_t code, uint8_t mode) {
    Action a; a.type = T_KEY; a.mods = mods; a.code = code; a.mode = mode; return a;
  };
  auto media = [](int32_t code, uint8_t mode) {
    Action a; a.type = T_MEDIA; a.code = code; a.mode = mode; return a;
  };
  cfg.map[IN_K1]       = key(0x01, 0x06, M_HOLD);   // Ctrl+C
  cfg.map[IN_K2]       = key(0x01, 0x19, M_HOLD);   // Ctrl+V
  cfg.map[IN_K3]       = key(0x01, 0x1D, M_HOLD);   // Ctrl+Z
  cfg.map[IN_K4]       = media(0x00CD, M_TAP);      // Play/Pause
  cfg.map[IN_K5]       = media(0x00B5, M_TAP);      // Next track
  cfg.map[IN_ENC_PUSH] = media(0x00E2, M_TAP);      // Mute
  cfg.map[IN_ENC_CW]   = media(0x00E9, M_TAP);      // Volume up
  cfg.map[IN_ENC_CCW]  = media(0x00EA, M_TAP);      // Volume down
  cfg.map[IN_PALM]     = media(0x00E2, M_BOTH);     // Mute, flipped while held
  cfg.map[IN_SWIPE_LR] = key(0x01, 0x1D, M_TAP);    // Ctrl+Z  (undo)
  cfg.map[IN_SWIPE_RL] = key(0x01, 0x1C, M_TAP);    // Ctrl+Y  (redo)
  for (int s = 0; s < mlCount; s++) {
    const char* lbl = IN_NAMES[IN_ML0 + s] + 3;
    if (!strcmp(lbl, "peace"))     cfg.map[IN_ML0 + s] = key(0x08, 0x0F, M_TAP);  // Win+L
    else if (!strcmp(lbl, "palm")) cfg.map[IN_ML0 + s] = key(0x08, 0x07, M_TAP);  // Win+D
  }
}

static uint8_t lookup(const char* s, const char* const* names, uint8_t n, uint8_t def) {
  for (uint8_t i = 0; i < n; i++) if (strcmp(s, names[i]) == 0) return i;
  return def;
}

static void configToJson(JsonObject o) {
  JsonObject m = o["map"].to<JsonObject>();
  for (int i = 0; i < inCount; i++) {
    JsonObject a = m[IN_NAMES[i]].to<JsonObject>();
    a["t"] = TYPE_NAMES[cfg.map[i].type];
    a["m"] = cfg.map[i].mods;
    a["c"] = cfg.map[i].code;
    a["mode"] = MODE_NAMES[cfg.map[i].mode];
  }
  JsonObject t = o["tune"].to<JsonObject>();
#define X(ty, n, d) t[#n] = cfg.tune.n;
  TUNE_FIELDS
#undef X
}

static void configFromJson(JsonObjectConst o) {
  JsonObjectConst m = o["map"];
  if (!m.isNull()) {
    for (int i = 0; i < inCount; i++) {
      JsonObjectConst a = m[IN_NAMES[i]];
      if (a.isNull()) continue;
      Action act;
      act.type = lookup(a["t"] | "none", TYPE_NAMES, 5, T_NONE);
      act.mods = a["m"] | 0;
      act.code = a["c"] | 0;
      act.mode = lookup(a["mode"] | "tap", MODE_NAMES, 3, M_TAP);
      cfg.map[i] = act;
    }
  }
  JsonObjectConst t = o["tune"];
  if (!t.isNull()) {
    int oldRoi = cfg.tune.roi, oldLock = cfg.tune.lockExposure, oldRot = cfg.tune.camRotate;
#define X(ty, n, d) cfg.tune.n = t[#n] | cfg.tune.n;
    TUNE_FIELDS
#undef X
    cfg.tune.roi = constrain(cfg.tune.roi, 4, 16);
    cfg.tune.streamFps = constrain(cfg.tune.streamFps, 1, 25);
    cfg.tune.encSteps = constrain(cfg.tune.encSteps, 1, 4);
    cfg.tune.mlFrames = constrain(cfg.tune.mlFrames, 1, 20);
    cfg.tune.mlRelease = constrain(cfg.tune.mlRelease, 1, 20);
    if (cfg.tune.roi != oldRoi) tofReconfig = true;
    if (cfg.tune.lockExposure != oldLock || cfg.tune.camRotate != oldRot) camReconfig = true;
  }
}

static void saveConfig() {
  JsonDocument d;
  configToJson(d.to<JsonObject>());
  String s;
  serializeJson(d, s);
  prefs.putString("cfg", s);
}

static void loadConfig() {
  setDefaults();
  String s = prefs.getString("cfg", "");
  if (s.length() == 0) return;
  JsonDocument d;
  if (deserializeJson(d, s) == DeserializationError::Ok) configFromJson(d.as<JsonObjectConst>());
}

static void sendConfig(const char* type = "config") {
  JsonDocument d;
  d["type"] = type;
  configToJson(d["config"].to<JsonObject>());
  sendJson(d);
}

// ============ HID OUTPUT ============
static void releaseEverything() {
  Keyboard.releaseAll();
  Consumer.release();
  Mouse.release(0x1F);
}

static void actPress(const Action& a) {
  switch (a.type) {
    case T_KEY:
      for (int b = 0; b < 8; b++) if (a.mods & (1 << b)) Keyboard.pressRaw(0xE0 + b);
      if (a.code) Keyboard.pressRaw((uint8_t)a.code);
      break;
    case T_MEDIA: Consumer.press((uint16_t)a.code); break;
    case T_MOUSE: Mouse.press((uint8_t)a.code); break;
    case T_WHEEL: Mouse.move(0, 0, (int8_t)constrain(a.code, -127, 127)); break;
  }
}

static void actRelease(const Action& a) {
  switch (a.type) {
    case T_KEY:
      if (a.code) Keyboard.releaseRaw((uint8_t)a.code);
      for (int b = 0; b < 8; b++) if (a.mods & (1 << b)) Keyboard.releaseRaw(0xE0 + b);
      break;
    case T_MEDIA: Consumer.release(); break;
    case T_MOUSE: Mouse.release((uint8_t)a.code); break;
    default: break;
  }
}

static void actTap(const Action& a) {
  actPress(a);
  delay(15);
  actRelease(a);
}

static uint32_t ledPulseUntil = 0;
static uint8_t heldCount = 0;

static void handleInput(uint8_t id, uint8_t phase) {
  if (id >= inCount) return;
  const Action& a = cfg.map[id];

  if (hostAttached) {
    JsonDocument d;
    d["type"] = "input";
    d["id"] = IN_NAMES[id];
    d["phase"] = phase == PH_TAP ? "tap" : phase == PH_DOWN ? "down" : "up";
    sendJson(d);
  }

  if (phase == PH_DOWN) heldCount++;
  if (phase == PH_UP && heldCount) heldCount--;
  ledPulseUntil = millis() + 80;

  if (a.type == T_NONE) return;
  switch (phase) {
    case PH_TAP:  actTap(a); break;
    case PH_DOWN: if (a.mode == M_HOLD) actPress(a); else actTap(a); break;
    case PH_UP:
      if (a.mode == M_HOLD) actRelease(a);
      else if (a.mode == M_BOTH) actTap(a);
      break;
  }
}

// ============ CAMERA ============
static bool initCameraFmt(pixformat_t fmt) {
  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0;
  c.ledc_timer = LEDC_TIMER_0;
  c.pin_d0 = 15; c.pin_d1 = 17; c.pin_d2 = 18; c.pin_d3 = 16;
  c.pin_d4 = 14; c.pin_d5 = 12; c.pin_d6 = 11; c.pin_d7 = 48;
  c.pin_xclk = 10; c.pin_pclk = 13; c.pin_vsync = 38; c.pin_href = 47;
  c.pin_sccb_sda = 40; c.pin_sccb_scl = 39;
  c.pin_pwdn = -1; c.pin_reset = -1;
  c.xclk_freq_hz = 20000000;
  c.pixel_format = fmt;
  c.frame_size = FRAMESIZE_QQVGA;  // 160x120
  c.jpeg_quality = 12;
  c.fb_count = 2;
  c.fb_location = CAMERA_FB_IN_PSRAM;
  c.grab_mode = CAMERA_GRAB_LATEST;
  esp_err_t e = esp_camera_init(&c);
  if (e != ESP_OK) camErr = e;
  return e == ESP_OK;
}

// Grayscale first; if the sensor refuses it, use YUV422 and keep only the luma bytes.
static bool initCamera() {
  if (initCameraFmt(PIXFORMAT_GRAYSCALE)) { camYuv = false; return true; }
  esp_camera_deinit();
  delay(50);
  if (initCameraFmt(PIXFORMAT_YUV422)) { camYuv = true; return true; }
  return false;
}

static void applyExposureLock(bool lock) {
  sensor_t* s = esp_camera_sensor_get();
  if (!s) return;
  s->set_gain_ctrl(s, lock ? 0 : 1);
  s->set_exposure_ctrl(s, lock ? 0 : 1);
}

// 180 deg rotation = vertical flip + horizontal mirror, done by the sensor itself, so the
// stream, the swipe tracker and the model all see the hand upright.
static void applyOrientation(bool rotate) {
  sensor_t* s = esp_camera_sensor_get();
  if (!s) return;
  s->set_vflip(s, rotate ? 1 : 0);
  s->set_hmirror(s, rotate ? 1 : 0);
}

// ============ ToF ============
static void configureTof() {
  tof.stopContinuous();
  tof.setDistanceMode(VL53L1X::Short);
  tof.setMeasurementTimingBudget(33000);
#if TOF_HAS_ROI
  tof.setROISize(cfg.tune.roi, cfg.tune.roi);
#endif
  tof.startContinuous(33);
}

// ============ VISION TASK (core 0): ToF + palm cover + swipes + stream ============
static float muClose(int d) {
  const Tune& t = cfg.tune;
  if (d <= t.tofFull) return 1.f;
  if (d >= t.tofZero) return 0.f;
  return (float)(t.tofZero - d) / (float)(t.tofZero - t.tofFull);
}

static float muCover(float c) {
  const Tune& t = cfg.tune;
  if (c <= t.covLow) return 0.f;
  if (c >= t.covHigh) return 1.f;
  return (c - t.covLow) / (t.covHigh - t.covLow);
}

static void sendReject(const char* id, const char* reason, float value) {
  if (!hostAttached) return;
  JsonDocument d;
  d["type"] = "reject";
  d["id"] = id;
  d["reason"] = reason;
  d["value"] = value;
  sendJson(d);
}

static void visionTask(void*) {
  static uint8_t full[FN];
  static uint8_t small[SN];
  static float bg[SN];
  static uint8_t framePkt[4 + FN];
  bool bgInit = false;

  // ToF filtering state
  int tofBuf[5] = {0};
  int tofCount = 0, tofIdx = 0, tofInvalid = 0;
  int tofDist = 9999, tofRaw = -1, tofStatus = -1;

  // Motion window (swipe) state
  bool inWin = false, tainted = false, armed = false;
  uint32_t winStart = 0;
  float x0 = 0.5f, xLast = 0.5f, peakConf = 0.f, peakCov = 0.f;
  int quiet = 0;

  // Palm state
  uint32_t darkSince = 0;
  uint32_t cooldownUntil = 0;

  // Camera lock
  uint32_t camStart = millis();
  int appliedLock = -1, appliedRot = -1;

  uint32_t lastStream = 0, fpsT0 = millis();
  int fpsFrames = 0;
  float fps = 0;

  for (;;) {
    const Tune& t = cfg.tune;
    uint32_t now = millis();

    if (tofOk && tofReconfig) { tofReconfig = false; configureTof(); }
    if (appliedRot != t.camRotate) {
      applyOrientation(t.camRotate);
      appliedRot = t.camRotate;
      bgInit = false;  // background is learned in image coordinates
    }
    if (now - camStart > 1500 && (appliedLock != t.lockExposure || camReconfig)) {
      camReconfig = false;
      applyExposureLock(t.lockExposure);
      appliedLock = t.lockExposure;
      bgInit = false;  // re-learn background after exposure change
    }

    // ---- ToF: drop invalid samples, median of last 5 valid ----
    if (tofOk && tof.dataReady()) {
      tof.read(false);
      tofStatus = tof.ranging_data.range_status;
      tofRaw = tof.ranging_data.range_mm;
      // 0 = valid, 3 = valid but clipped at minimum range (hand very close)
      if (tofStatus == 0 || tofStatus == 3) {
        tofInvalid = 0;
        tofBuf[tofIdx] = tofRaw + t.tofOffset;
        tofIdx = (tofIdx + 1) % 5;
        if (tofCount < 5) tofCount++;
        int s[5];
        for (int i = 0; i < tofCount; i++) s[i] = tofBuf[i];
        for (int i = 1; i < tofCount; i++) {
          int v = s[i], j = i - 1;
          while (j >= 0 && s[j] > v) { s[j + 1] = s[j]; j--; }
          s[j + 1] = v;
        }
        tofDist = s[tofCount / 2];
      } else {
        ++tofInvalid;
        // A palm right on the sensor is closer than it can measure (~4 cm), so it reports
        // errors instead of a distance. If the hand was already near, or the sensor says
        // "min range fail" (13), treat it as touching instead of "nothing there".
        // Held only briefly (or for as long as the palm cover is active), so errors with
        // nothing in front of the sensor still end up as "far".
        bool wasNear = tofDist <= t.tofZero;
        if (tofStatus == 13 || (wasNear && tofInvalid >= 2 && tofInvalid <= 10)) {
          tofDist = 0;
          tofCount = 0;
        } else if (tofInvalid > 10 && !palmActive) {
          tofDist = 9999;   // nothing in range: treat as far
          tofCount = 0;
        }
      }
      gTofDist = tofDist;
    }

    // ---- Camera ----
    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) { vTaskDelay(5); continue; }
    const size_t need = camYuv ? FN * 2 : FN;
    if (fb->width != FW || fb->height != FH || fb->len < need) {
      camBadW = fb->width; camBadH = fb->height; camBadLen = fb->len;
      esp_camera_fb_return(fb); vTaskDelay(5); continue;
    }
    if (camYuv) { for (int i = 0; i < FN; i++) full[i] = fb->buf[i * 2]; }   // YUYV: Y on even bytes
    else memcpy(full, fb->buf, FN);
    esp_camera_fb_return(fb);
    camFrames++;

#if USE_ML
    // Hand the newest frame to the model when it's ready for one
    if (t.mlOn && mlCount && !mlFrameReady) {
      memcpy(mlFrame, full, FN);
      mlFrameReady = true;
      xSemaphoreGive(mlSem);
    }
#endif

    for (int y = 0; y < SH; y++) {
      const uint8_t* r0 = full + (y * 2) * FW;
      const uint8_t* r1 = r0 + FW;
      for (int x = 0; x < SW; x++) {
        int sx = x * 2;
        small[y * SW + x] = (r0[sx] + r0[sx + 1] + r1[sx] + r1[sx + 1]) >> 2;
      }
    }

    uint32_t sum = 0;
    for (int i = 0; i < SN; i++) sum += small[i];
    float meanF = (float)sum / SN;

    if (!bgInit) {
      for (int i = 0; i < SN; i++) bg[i] = small[i];
      bgInit = true;
    }
    float bgSum = 0;
    for (int i = 0; i < SN; i++) bgSum += bg[i];
    float bgMean = max(1.f, bgSum / SN);

    // Foreground: pixels that differ from the learned background
    int cnt = 0;
    float sx = 0;
    for (int y = 0; y < SH; y++)
      for (int x = 0; x < SW; x++) {
        int i = y * SW + x;
        if (fabsf((float)small[i] - bg[i]) > t.diffThr) { cnt++; sx += x; }
      }
    float cov = (float)cnt / SN;
    float cx = cnt ? (sx / cnt) / (SW - 1) : 0.5f;
    if (t.mirror) cx = 1.f - cx;

    float mc = tofOk ? muClose(tofDist) : 1.f;   // no ToF -> don't block gestures
    float mv = muCover(cov);
    float conf = min(mc, mv);                    // fuzzy AND

    // ---- Palm cover ----
    float ratio = meanF / bgMean;
    if (!palmActive) {
      if (ratio < t.palmDark && mc > 0.3f && now >= cooldownUntil) {
        if (!darkSince) darkSince = now;
        if (now - darkSince >= (uint32_t)t.palmHoldMs) {
          palmActive = true;
          tainted = true;
          postInput(IN_PALM, PH_DOWN);
        }
      } else darkSince = 0;
    } else if (ratio > t.palmExit) {
      palmActive = false;
      darkSince = 0;
      cooldownUntil = now + t.cooldownMs;
      postInput(IN_PALM, PH_UP);
    }

    // ---- Swipe motion window ----
    bool moving = cov > t.trackCov;
    if (!moving) armed = true;
#if USE_ML
    bool poseHeld = mlActiveSlot >= 0;           // a held pose isn't a swipe
#else
    bool poseHeld = false;
#endif
    if (!inWin && moving && armed && !palmActive && !poseHeld && now >= cooldownUntil) {
      armed = false;
      inWin = true; tainted = false; winStart = now;
      x0 = xLast = cx; peakConf = conf; peakCov = cov; quiet = 0;
      gWinStart = now;
    }
    if (inWin) {
      if (moving) { xLast = cx; quiet = 0; } else quiet++;
      peakConf = max(peakConf, conf);
      peakCov = max(peakCov, cov);
      if (palmActive || poseHeld) tainted = true;

      if (now - winStart > 5000 && !palmActive && !poseHeld) {
        // Something parked in view: re-learn background instead of waiting forever
        inWin = false;
        for (int i = 0; i < SN; i++) bg[i] = small[i];
      } else if (quiet >= 2) {
        inWin = false;
        uint32_t dt = now - winStart;
        float travel = xLast - x0;
        if (!tainted) {
          if (peakConf < t.confOn) {
            if (peakCov > t.covLow * 0.5f) sendReject("swipe", "low_conf", peakConf);
          } else if (dt > (uint32_t)t.swipeMaxMs) {
            sendReject("swipe", "too_slow", dt);
          } else if (fabsf(travel) < t.swipeTravel) {
            sendReject("swipe", "short_travel", fabsf(travel));
          } else {
            postInput(travel > 0 ? IN_SWIPE_LR : IN_SWIPE_RL, PH_TAP);
            cooldownUntil = now + t.cooldownMs;
          }
        }
      }
    }
    swipeWindow = inWin;

    // Background learns only while nothing is happening
    if (!inWin && !palmActive && !poseHeld) {
      const float a = 0.05f;
      for (int i = 0; i < SN; i++) bg[i] += a * ((float)small[i] - bg[i]);
    }

    // ---- FPS ----
    fpsFrames++;
    if (now - fpsT0 >= 1000) { fps = fpsFrames * 1000.f / (now - fpsT0); fpsFrames = 0; fpsT0 = now; }

    // ---- Debug stream ----
    if (hostAttached && streamOn && now - lastStream >= (uint32_t)(1000 / t.streamFps)) {
      lastStream = now;
      framePkt[0] = FW & 0xFF; framePkt[1] = FW >> 8;
      framePkt[2] = FH & 0xFF; framePkt[3] = FH >> 8;
      memcpy(framePkt + 4, full, FN);
      sendPacket(1, framePkt, sizeof(framePkt));

      JsonDocument d;
      d["type"] = "tel";
      d["tof"] = tofOk ? tofDist : -1;
      d["raw"] = tofRaw;
      d["st"] = tofStatus;
      d["cov"] = cov;
      d["cx"] = cx;
      d["mc"] = mc;
      d["mv"] = mv;
      d["conf"] = conf;
      d["ratio"] = ratio;
      d["palm"] = (bool)palmActive;
      d["win"] = inWin;
      d["fps"] = fps;
#if USE_ML
      if (mlCount) {
        JsonArray ml = d["ml"].to<JsonArray>();
        for (int i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) ml.add(mlScores[i]);
        d["mlms"] = mlMs;
        d["mlOn"] = (bool)t.mlOn;
      }
#endif
      sendJson(d);
    }
  }
}

// ============ ML TASK (core 1): Edge Impulse pose classifier ============
#if USE_ML
static int mlGetData(size_t offset, size_t length, float* out) {
  for (size_t i = 0; i < length; i++) {
    uint32_t g = mlIn[offset + i];
    out[i] = (float)((g << 16) | (g << 8) | g);   // packed RGB; model converts to grayscale
  }
  return 0;
}

// Center-crop to a square and bilinear-resize to the model input
// (matches Edge Impulse "Fit shortest axis").
static void prepareModelInput(const uint8_t* src) {
  const int OW = EI_CLASSIFIER_INPUT_WIDTH, OH = EI_CLASSIFIER_INPUT_HEIGHT;
  const float srcAspect = (float)FW / FH, dstAspect = (float)OW / OH;
  float cw = FW, ch = FH;
  if (srcAspect > dstAspect) cw = FH * dstAspect; else ch = FW / dstAspect;
  const float ox = (FW - cw) * 0.5f, oy = (FH - ch) * 0.5f;
  const float scx = cw / OW, scy = ch / OH;
  for (int y = 0; y < OH; y++) {
    float fy = oy + (y + 0.5f) * scy - 0.5f;
    int y0 = constrain((int)fy, 0, FH - 1), y1 = min(y0 + 1, FH - 1);
    float wy = fy - y0;
    for (int x = 0; x < OW; x++) {
      float fx = ox + (x + 0.5f) * scx - 0.5f;
      int x0 = constrain((int)fx, 0, FW - 1), x1 = min(x0 + 1, FW - 1);
      float wx = fx - x0;
      float top = src[y0 * FW + x0] * (1 - wx) + src[y0 * FW + x1] * wx;
      float bot = src[y1 * FW + x0] * (1 - wx) + src[y1 * FW + x1] * wx;
      mlIn[y * OW + x] = (uint8_t)(top * (1 - wy) + bot * wy + 0.5f);
    }
  }
}

static int modelIdxToSlot(int idx) {
  for (int s = 0; s < mlCount; s++) if (mlModelIdx[s] == idx) return s;
  return -1;
}

static void mlTask(void*) {
  int streak = 0, lastIdx = -1, lost = 0, blocked = 0;
  bool needRelease = false;
  uint32_t cooldownUntil = 0;

  for (;;) {
    if (xSemaphoreTake(mlSem, pdMS_TO_TICKS(200)) != pdTRUE) {
      // Model switched off or no frames: end any held pose
      if (!cfg.tune.mlOn && mlActiveSlot >= 0) {
        postInput(IN_ML0 + mlActiveSlot, PH_UP);
        mlActiveSlot = -1;
      }
      continue;
    }
    prepareModelInput(mlFrame);
    mlFrameReady = false;                 // vision task may hand over the next frame now

    signal_t signal;
    signal.total_length = EI_CLASSIFIER_INPUT_WIDTH * EI_CLASSIFIER_INPUT_HEIGHT;
    signal.get_data = &mlGetData;
    ei_impulse_result_t result = {};
    uint32_t t0 = millis();
    if (run_classifier(&signal, &result, false) != EI_IMPULSE_OK) continue;
    mlMs = millis() - t0;

    int best = 0;
    for (int i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) {
      mlScores[i] = result.classification[i].value;
      if (result.classification[i].value > result.classification[best].value) best = i;
    }

    const Tune& t = cfg.tune;
    const uint32_t now = millis();
    const float score = result.classification[best].value;
    const bool confident = best != mlNoneIdx && score >= t.mlThr;
    // A pose only counts once the hand has been in view longer than a swipe can last,
    // so a hand passing through for a swipe never triggers one.
    const bool settled = !swipeWindow || now - gWinStart > (uint32_t)t.swipeMaxMs;
    const bool distOk = t.mlMaxDist <= 0 || !tofOk || gTofDist <= t.mlMaxDist;
    const bool gateOk = !palmActive && settled && distOk;

    // A pose is being held: keep it until it's gone for mlRelease frames
    if (mlActiveSlot >= 0) {
      if (confident && best == mlModelIdx[mlActiveSlot] && !palmActive) lost = 0;
      else if (++lost >= t.mlRelease) {
        postInput(IN_ML0 + mlActiveSlot, PH_UP);
        mlActiveSlot = -1;
        lost = 0;
        needRelease = true;
        cooldownUntil = now + t.mlCooldownMs;
      }
      continue;
    }

    if (!confident) { streak = 0; lastIdx = -1; blocked = 0; needRelease = false; continue; }
    if (needRelease || now < cooldownUntil) continue;

    if (!gateOk) {
      streak = 0;
      // Report once per attempt why a confident pose was ignored
      if (settled && !distOk && ++blocked == t.mlFrames) {
        char id[24];
        snprintf(id, sizeof(id), "ml_%s", ei_classifier_inferencing_categories[best]);
        sendReject(id, "too_far", gTofDist >= 9999 ? -1 : gTofDist);
      }
      continue;
    }
    blocked = 0;

    streak = (best == lastIdx) ? streak + 1 : 1;
    lastIdx = best;
    if (streak >= t.mlFrames) {
      int slot = modelIdxToSlot(best);
      if (slot >= 0) {
        mlActiveSlot = slot;
        lost = 0;
        postInput(IN_ML0 + slot, PH_DOWN);
      }
      streak = 0;
    }
  }
}
#endif

// ============ KEYS + ENCODER ============
static bool keyState[2][3] = {};
static uint32_t keyChanged[2][3] = {};

static void scanMatrix() {
  uint32_t now = millis();
  for (int r = 0; r < 2; r++) {
    digitalWrite(ROW_PINS[r], LOW);
    delayMicroseconds(10);
    for (int c = 0; c < 3; c++) {
      bool pressed = digitalRead(COL_PINS[c]) == LOW;
      if (pressed != keyState[r][c] && now - keyChanged[r][c] > 5) {
        keyState[r][c] = pressed;
        keyChanged[r][c] = now;
        handleInput(MATRIX_MAP[r][c], pressed ? PH_DOWN : PH_UP);
      }
    }
    digitalWrite(ROW_PINS[r], HIGH);
  }
}

static uint8_t encPrev = 0;
static int encAcc = 0;

static void pollEncoder() {
  static const int8_t table[16] = {0, -1, 1, 0, 1, 0, 0, -1, -1, 0, 0, 1, 0, 1, -1, 0};
  uint8_t s = (digitalRead(ENC_A_PIN) << 1) | digitalRead(ENC_B_PIN);
  encPrev = ((encPrev << 2) | s) & 0x0F;
  encAcc += table[encPrev];
  int steps = cfg.tune.encSteps;
  if (encAcc >= steps || encAcc <= -steps) {
    bool cw = (encAcc > 0) ^ (bool)cfg.tune.encReverse;
    encAcc = 0;
    handleInput(cw ? IN_ENC_CW : IN_ENC_CCW, PH_TAP);
  }
}

// ============ COMMANDS FROM CONFIGURATOR ============
// Newline-terminated JSON: {"cmd":"hello"} | get_config | set_config {"config":{...}}
//                          | save | defaults | stream {"on":true} | bye
static void logText(const char* msg);
static void handleLine(const String& line) {
  JsonDocument d;
  if (deserializeJson(d, line) != DeserializationError::Ok) return;
  const char* cmd = d["cmd"] | "";
  if (!hostAttached && strcmp(cmd, "hello")) { char m[64]; snprintf(m, sizeof(m), "got command '%s'", cmd); logText(m); }

  if (!strcmp(cmd, "hello")) {
    hostAttached = true;
    JsonDocument r;
    r["type"] = "hello";
    r["fw"] = FW_VERSION;
    r["w"] = FW;
    r["h"] = FH;
    r["tof"] = tofOk;
    r["cam"] = camOk;
    r["camErr"] = camErr;
    r["camFmt"] = camYuv ? "yuv" : "gray";
    r["psram"] = (uint32_t)ESP.getPsramSize();
#if USE_ML
    JsonArray labels = r["labels"].to<JsonArray>();
    for (int i = 0; i < EI_CLASSIFIER_LABEL_COUNT; i++) labels.add(ei_classifier_inferencing_categories[i]);
    r["mlIn"] = EI_CLASSIFIER_INPUT_WIDTH;
#endif
    sendJson(r);
    sendConfig();
  } else if (!strcmp(cmd, "bye")) {
    streamOn = false;
    hostAttached = false;
  } else if (!strcmp(cmd, "get_config")) {
    sendConfig();
  } else if (!strcmp(cmd, "set_config")) {
    releaseEverything();
    configFromJson(d["config"].as<JsonObjectConst>());
  } else if (!strcmp(cmd, "save")) {
    saveConfig();
    JsonDocument r;
    r["type"] = "saved";
    sendJson(r);
  } else if (!strcmp(cmd, "defaults")) {
    releaseEverything();
    setDefaults();
    tofReconfig = camReconfig = true;
    sendConfig();
  } else if (!strcmp(cmd, "stream")) {
    streamOn = d["on"] | false;
  }
}

static String lineBuf;

static void readLink() {
  while (Link.available()) {
    int ch = Link.read();
    if (ch == '\n') { handleLine(lineBuf); lineBuf = ""; }
    else if (ch != '\r' && lineBuf.length() < 4096) lineBuf += (char)ch;
  }
}

// ============ SETUP / LOOP ============
static void blink(int n) {
  for (int i = 0; i < n; i++) { digitalWrite(LED_PIN, HIGH); delay(150); digitalWrite(LED_PIN, LOW); delay(150); }
}

void setup() {
  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, HIGH);
  for (auto p : ROW_PINS) { pinMode(p, OUTPUT); digitalWrite(p, HIGH); }
  for (auto p : COL_PINS) pinMode(p, INPUT_PULLUP);
  pinMode(ENC_A_PIN, INPUT_PULLUP);
  pinMode(ENC_B_PIN, INPUT_PULLUP);

  linkMtx = xSemaphoreCreateMutex();
  inputQ = xQueueCreate(32, sizeof(InputEvt));

  setupModelInputs();
  prefs.begin("gpad", false);
  loadConfig();

  Keyboard.begin();
  Consumer.begin();
  Mouse.begin();
  Link.begin();
  USB.productName("Gesture Pad");
  USB.begin();

  Wire.begin(TOF_SDA, TOF_SCL);
  Wire.setClock(400000);
  tof.setTimeout(100);
  tofOk = tof.init();
  if (tofOk) configureTof();

  bool mlOk = true;
#if USE_ML
  mlSem = xSemaphoreCreateBinary();
  mlFrame = (uint8_t*)ps_malloc(FN);
  mlOk = mlFrame != nullptr && psramFound();
  if (mlOk && mlCount) xTaskCreatePinnedToCore(mlTask, "ml", 16384, nullptr, 1, nullptr, 1);
#endif

  camOk = initCamera();
  if (camOk) xTaskCreatePinnedToCore(visionTask, "vision", 8192, nullptr, 2, nullptr, 0);

  // Keys/encoder run in loop(); give it priority over the model so they stay responsive.
  vTaskPrioritySet(nullptr, 2);

  // Status blinks: 3 = camera failed, 2 = ToF not found, 4 = no PSRAM (model can't run)
  digitalWrite(LED_PIN, LOW);
  delay(200);
  if (!camOk) blink(3);
  else if (!mlOk) blink(4);
  else if (!tofOk) blink(2);
}

// Plain-text status for the Arduino Serial Monitor (the app ignores text between packets)
static void logText(const char* msg) {
  if (hostAttached) return;
  xSemaphoreTake(linkMtx, portMAX_DELAY);
  Link.print("[gpad] ");
  Link.println(msg);
  xSemaphoreGive(linkMtx);
}

void loop() {
  // Port closed by the host (DTR dropped) -> stop streaming. Edge-triggered only.
  static bool wasOpen = false;
  bool open = (bool)Link;
  if (wasOpen && !open) { hostAttached = false; streamOn = false; }
  wasOpen = open;

  static uint32_t lastAlive = 0;
  if (!hostAttached && millis() - lastAlive >= 2000) {
    lastAlive = millis();
    char buf[200];
    snprintf(buf, sizeof(buf), "fw %s running | camera %s%s (err 0x%x, %lu frames) | tof %s | psram %lu KB | poses %d | waiting for the app",
             FW_VERSION, camOk ? "ok" : "FAILED", camYuv ? " yuv" : "", camErr, (unsigned long)camFrames,
             tofOk ? "ok" : "not found", (unsigned long)(ESP.getPsramSize() / 1024), mlCount);
    logText(buf);
  }
  readLink();

  static uint32_t lastScan = 0;
  if (millis() - lastScan >= 2) { lastScan = millis(); scanMatrix(); }
  pollEncoder();

  InputEvt e;
  while (xQueueReceive(inputQ, &e, 0) == pdTRUE) handleInput(e.id, e.phase);

  // Once a second while the app is watching: camera health, even if no frames arrive
  static uint32_t lastStatus = 0;
  if (hostAttached && streamOn && millis() - lastStatus >= 1000) {
    lastStatus = millis();
    JsonDocument d;
    d["type"] = "status";
    d["cam"] = camOk;
    d["camErr"] = camErr;
    d["frames"] = camFrames;
    if (camBadW || camBadLen) { d["badW"] = camBadW; d["badH"] = camBadH; d["badLen"] = camBadLen; }
    sendJson(d);
  }

  bool poseHeld = false;
#if USE_ML
  poseHeld = mlActiveSlot >= 0;
#endif
  digitalWrite(LED_PIN, (palmActive || poseHeld || heldCount || millis() < ledPulseUntil) ? HIGH : LOW);

  vTaskDelay(1);   // let the model task use the rest of core 1
}

