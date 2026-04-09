#include <glad/glad.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "style.h" // Custom window styling

#include "Inter-Medium.h"
#include "JetBrainsMono-Regular.h"

#ifdef __linux__
#include <X11/Xlib.h>
#include <X11/keysym.h>
#include <fcntl.h>
#include <signal.h>
#include <unistd.h>
#endif
#ifdef _WIN32
#include <windows.h>
#endif

#include <atomic>
#include <chrono>
#include <clocale>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "common.h"
#include "ocr_engine.h"
#include "sco.h"
#include "tinyfiledialogs.h"

// Default model paths set by CMake
#ifndef OHSEEARE_MODEL
#define OHSEEARE_MODEL "models/GLM-OCR-f16.gguf"
#endif
#ifndef OHSEEARE_MMPROJ
#define OHSEEARE_MMPROJ "models/mmproj-GLM-OCR-Q8_0.gguf"
#endif

#define INITIAL_PROMPT "Latexify"

static std::atomic<bool> capture_requested{false};
static std::atomic<bool> window_hide_requested{false};
static std::atomic<bool> app_quit_requested{false};

// Global hotkey listener  (Linux: Ctrl+Shift+S via XGrabKey)
// Should be configurable eventually

static void start_hotkey_listener(std::atomic<bool> *flag) {
#ifdef __linux__
  std::thread([flag] {
    Display *dpy = XOpenDisplay(nullptr);
    if (!dpy)
      return;
    Window root = DefaultRootWindow(dpy);

    KeyCode kc = XKeysymToKeycode(dpy, XK_s);
    const unsigned mods[] = {0u, (unsigned)Mod2Mask, (unsigned)LockMask,
                             (unsigned)(Mod2Mask | LockMask)};
    for (unsigned mod : mods)
      XGrabKey(dpy, kc, ControlMask | ShiftMask | mod, root, True,
               GrabModeAsync, GrabModeAsync);
    XSelectInput(dpy, root, KeyPressMask);

    while (true) {
      XEvent ev;
      XNextEvent(dpy, &ev);
      if (ev.type == KeyPress) {
        flag->store(true);
        glfwPostEmptyEvent(); // wake the main loop
      }
    }
  }).detach();
#elif defined(_WIN32)
  std::thread([flag] {
    if (!RegisterHotKey(nullptr, 1, MOD_CONTROL | MOD_SHIFT | MOD_NOREPEAT,
                        'S'))
      return;
    MSG msg;
    while (GetMessage(&msg, nullptr, 0, 0)) {
      if (msg.message == WM_HOTKEY && msg.wParam == 1) {
        flag->store(true);
        glfwPostEmptyEvent();
      }
    }
    UnregisterHotKey(nullptr, 1);
  }).detach();
#else
  (void)flag;
#endif
}
// Helpers
static const char *format_elapsed(double seconds, char *buf, size_t len) {
  if (seconds < 1.0) {
    snprintf(buf, len, "%.0f ms", seconds * 1000.0);
  } else if (seconds < 60.0) {
    snprintf(buf, len, "%.2f s", seconds);
  } else {
    int m = (int)(seconds / 60.0);
    double s = seconds - m * 60.0;
    snprintf(buf, len, "%dm %.1fs", m, s);
  }
  return buf;
}

static void center_window_on_primary(GLFWwindow *w) {
  GLFWmonitor *mon = glfwGetPrimaryMonitor();
  if (!mon) {
    return;
  }
  const GLFWvidmode *mode = glfwGetVideoMode(mon);
  if (!mode) {
    return;
  }
  int ww, wh;
  glfwGetWindowSize(w, &ww, &wh);
  int mx, my;
  glfwGetMonitorPos(mon, &mx, &my);
  glfwSetWindowPos(w, mx + (mode->width - ww) / 2,
                   my + (mode->height - wh) / 2);
}

struct AppSettings {
  std::string model_path = OHSEEARE_MODEL;
  std::string mmproj_path = OHSEEARE_MMPROJ;
  float temperature = 0.1f;
  int max_tokens = 2048;
  int n_threads = 8;
  bool auto_copy = false;
};

// GLFW close callback hide the popup instead of quitting
static void close_callback(GLFWwindow *w) {
  glfwSetWindowShouldClose(w, GLFW_FALSE);
  window_hide_requested.store(true);
}

int main(int argc, char **argv) {
  AppSettings cfg;
  cfg.model_path = OHSEEARE_MODEL;
  cfg.mmproj_path = OHSEEARE_MMPROJ;

  std::string &model_path_str = cfg.model_path;
  std::string &mmproj_path_str = cfg.mmproj_path;

  for (int i = 1; i < argc; i++) {
    if ((strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) &&
        i + 1 < argc) {
      model_path_str = argv[++i];
    } else if (strcmp(argv[i], "--mmproj") == 0 && i + 1 < argc) {
      mmproj_path_str = argv[++i];
    } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      fprintf(stderr,
              "Usage: ohseeare-gui [-m model.gguf] [--mmproj mmproj.gguf]\n");
      return 0;
    }
  }

#ifdef __linux__
  {
    pid_t pid = fork();
    if (pid < 0) {
      perror("fork");
      return 1;
    }
    if (pid > 0) {
      return 0; // parent: exit, returning the prompt to the shell
    }

    // child: redirect stdio to /dev/null and survive terminal hangup
    signal(SIGHUP, SIG_IGN);
    int fd = open("/dev/null", O_RDWR);
    if (fd >= 0) {
      dup2(fd, STDIN_FILENO);
      dup2(fd, STDOUT_FILENO);
      dup2(fd, STDERR_FILENO);
      close(fd);
    }
  }
#elif defined(_WIN32)
  FreeConsole();
#endif

  std::setlocale(LC_NUMERIC, "C");
  ggml_time_init();
  common_init();

  if (!glfwInit()) {
    return 1;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);      // start hidden
  glfwWindowHint(GLFW_FLOATING, GLFW_TRUE);      // always on top
  glfwWindowHint(GLFW_FOCUS_ON_SHOW, GLFW_TRUE); // grab focus when shown
  glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
  glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
  glfwWindowHint(GLFW_DECORATED, GLFW_FALSE); // no title bar, borders, etc.

  GLFWwindow *window = glfwCreateWindow(560, 460, "ohseeare", nullptr, nullptr);
  glfwSetWindowPos(window, 500, 500);
  if (!window) {
    glfwTerminate();
    return 1;
  }
  glfwDefaultWindowHints(); // reset so capture overlay isn't invisible
  glfwMakeContextCurrent(window);
  gladLoadGL();
  glfwSwapInterval(1);
  glfwSetWindowCloseCallback(window, close_callback);
  glfwSetWindowSizeLimits(window, 360, 280, GLFW_DONT_CARE, GLFW_DONT_CARE);

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();

  ImGuiIO &io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

  // Load fonts from embedded byte arrays because I don't want to depend on more
  // junk
  ImFont *font_mono = nullptr;
  {
    ImFontConfig cfg;
    cfg.FontDataOwnedByAtlas = false; // static arrays, don't free them

    io.Fonts->AddFontFromMemoryTTF(Inter_Medium_ttf, (int)Inter_Medium_ttf_len,
                                   15.0f, &cfg);
    font_mono = io.Fonts->AddFontFromMemoryTTF(
        JetBrainsMono_Regular_ttf, (int)JetBrainsMono_Regular_ttf_len, 14.0f,
        &cfg);

    io.Fonts->Build();
  }

  apply_custom_style();

  ImGui_ImplGlfw_InitForOpenGL(window, true);
  ImGui_ImplOpenGL3_Init("#version 330");

  static char prompt[512] = INITIAL_PROMPT;

  OcrEngine engine;
  fprintf(stderr, "\n  Loading model: %s\n  Loading mmproj: %s\n",
          model_path_str.c_str(), mmproj_path_str.c_str());
  if (!engine.load(model_path_str.c_str(), mmproj_path_str.c_str(),
                   cfg.n_threads)) {
    fprintf(stderr, "  ERROR: failed to load model or mmproj.\n");
    return 1;
  }
  fprintf(stderr, "  Model loaded successfully.\n");

  // Shared state
  std::string result;
  std::string status = "Ready";
  std::atomic<bool> ocr_running{false};
  std::atomic<bool> rerun_requested{false};
  std::shared_ptr<ScoImage> last_cap; // last successful capture, kept for rerun
  std::mutex result_mutex;
  std::thread ocr_thread;

  // Settings panel state
  bool settings_open = false;
  std::string settings_model_path = model_path_str;
  std::string settings_mmproj_path = mmproj_path_str;
  float settings_temperature = cfg.temperature;
  int settings_max_tokens = cfg.max_tokens;
  int settings_n_threads = cfg.n_threads;
  bool settings_auto_copy = cfg.auto_copy;
  std::atomic<bool> model_reload_requested{false};
  std::atomic<bool> model_reloading{false};
  std::string model_reload_status; // protected by result_mutex
  std::thread reload_thread;
  std::atomic<bool> auto_copy_pending{false};

  // Timer state
  using Clock = std::chrono::steady_clock;
  Clock::time_point timer_start;
  double last_elapsed_sec = 0.0;
  std::atomic<bool> timer_active{false};

  // Window visibility
  bool window_visible = false;

  // Debounce Escape so a single press doesn't re-trigger
  bool escape_was_pressed = false;

  start_hotkey_listener(&capture_requested);

  fprintf(stderr, "\n"
                  "  ohseeare running in background.\n"
                  "  Press Ctrl+Shift+S to capture a screen region.\n"
                  "\n");

  while (!app_quit_requested.load()) {
    // When hidden, sleep efficiently; when visible, poll normally.
    if (window_visible) {
      glfwPollEvents();
    } else {
      glfwWaitEventsTimeout(0.1);
    }

    if (window_hide_requested.exchange(false) && window_visible) {
      glfwHideWindow(window);
      window_visible = false;
    }

    if (window_visible) {
      bool esc_down = glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS;
      if (esc_down && !escape_was_pressed) {
        glfwHideWindow(window);
        window_visible = false;
      }
      escape_was_pressed = esc_down;
    } else {
      escape_was_pressed = false;
    }

    if (capture_requested.exchange(false) && !ocr_running.load()) {
      // Hide the popup if it's showing so it doesn't appear in the screenshot
      bool was_visible = window_visible;
      if (was_visible) {
        glfwHideWindow(window);
        window_visible = false;
      }
      // Let the compositor/WM finish hiding
      for (int i = 0; i < 3; i++) {
        glfwPollEvents();
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(150));

      ScoImage cap = sco_capture_region();
      bool captured = !cap.rgba.empty();

      if (captured) {
        // Show the popup and start OCR
        center_window_on_primary(window);
        glfwShowWindow(window);
        glfwFocusWindow(window);
        window_visible = true;

        if (ocr_thread.joinable()) {
          ocr_thread.join();
        }

        ocr_running.store(true);
        timer_start = Clock::now();
        timer_active = true;
        {
          std::lock_guard<std::mutex> lock(result_mutex);
          status = "Running OCR\xe2\x80\xa6";
          result.clear();
        }

        last_cap = std::make_shared<ScoImage>(std::move(cap));

        std::string prompt_copy(prompt);
        float infer_temp = settings_temperature;
        int infer_max_tokens = settings_max_tokens;
        bool infer_auto_copy = settings_auto_copy;
        ocr_thread = std::thread([&engine, &result, &status, &result_mutex,
                                  &ocr_running, &last_elapsed_sec,
                                  &timer_active, &timer_start,
                                  &auto_copy_pending, prompt_copy,
                                  cap_ptr = last_cap, infer_temp,
                                  infer_max_tokens, infer_auto_copy] {
          std::string text = engine.infer(cap_ptr->rgba.data(), cap_ptr->width,
                                          cap_ptr->height, prompt_copy.c_str(),
                                          infer_temp, infer_max_tokens);

          double elapsed =
              std::chrono::duration<double>(Clock::now() - timer_start).count();

          {
            std::lock_guard<std::mutex> lock(result_mutex);
            result = std::move(text);
            last_elapsed_sec = elapsed;
            timer_active = false;

            char tbuf[64];
            format_elapsed(elapsed, tbuf, sizeof(tbuf));
            status = std::string("Done in ") + tbuf;
          }
          if (infer_auto_copy)
            auto_copy_pending.store(true);
          ocr_running.store(false);
        });
      } else {
        // Capture was cancelled
        if (was_visible) {
          // Restore the popup if it was showing before
          glfwShowWindow(window);
          glfwFocusWindow(window);
          window_visible = true;
        }
        std::lock_guard<std::mutex> lock(result_mutex);
        status = "Capture cancelled.";
        timer_active = false;
      }
    }

    // Handle rerun request
    if (rerun_requested.exchange(false) && !ocr_running.load() && last_cap) {
      if (ocr_thread.joinable()) {
        ocr_thread.join();
      }

      ocr_running.store(true);
      timer_start = Clock::now();
      timer_active = true;
      {
        std::lock_guard<std::mutex> lock(result_mutex);
        status = "Running OCR\xe2\x80\xa6";
        result.clear();
      }

      std::string prompt_copy(prompt);
      float infer_temp = settings_temperature;
      int infer_max_tokens = settings_max_tokens;
      bool infer_auto_copy = settings_auto_copy;

      ocr_thread = std::thread([&engine, &result, &status, &result_mutex,
                                &ocr_running, &last_elapsed_sec, &timer_active,
                                &timer_start, &auto_copy_pending, prompt_copy,
                                cap_ptr = last_cap, infer_temp,
                                infer_max_tokens, infer_auto_copy] {
        std::string text =
            engine.infer(cap_ptr->rgba.data(), cap_ptr->width, cap_ptr->height,
                         prompt_copy.c_str(), infer_temp, infer_max_tokens);

        double elapsed =
            std::chrono::duration<double>(Clock::now() - timer_start).count();

        {
          std::lock_guard<std::mutex> lock(result_mutex);
          result = std::move(text);
          last_elapsed_sec = elapsed;
          timer_active = false;

          char tbuf[64];
          format_elapsed(elapsed, tbuf, sizeof(tbuf));
          status = std::string("Done in ") + tbuf;
        }
        if (infer_auto_copy) {
          auto_copy_pending.store(true);
        }
        ocr_running.store(false);
      });
    }

    // Handle auto-copy after OCR
    if (auto_copy_pending.exchange(false)) {
      std::lock_guard<std::mutex> lock(result_mutex);
      ImGui::SetClipboardText(result.c_str());
      status = "Done \xe2\x80\x94 copied to clipboard!";
    }

    // Handle model reload request
    if (model_reload_requested.exchange(false) && !model_reloading.load() &&
        !ocr_running.load()) {
      std::string new_model = settings_model_path;
      std::string new_mmproj = settings_mmproj_path;
      int new_n_threads = settings_n_threads;
      if (reload_thread.joinable())
        reload_thread.join();
      model_reloading.store(true);
      {
        std::lock_guard<std::mutex> lk(result_mutex);
        model_reload_status.clear();
      }
      reload_thread =
          std::thread([&engine, &model_path_str, &mmproj_path_str, &cfg,
                       &model_reloading, &result_mutex, &model_reload_status,
                       new_model, new_mmproj, new_n_threads]() {
            engine.free();
            bool ok = engine.load(new_model.c_str(), new_mmproj.c_str(),
                                  new_n_threads);
            std::lock_guard<std::mutex> lk(result_mutex);
            if (ok) {
              model_path_str = new_model;
              mmproj_path_str = new_mmproj;
              cfg.n_threads = new_n_threads;
              model_reload_status = "Model reloaded successfully.";
            } else {
              model_reload_status = "ERROR: failed to load model. Check paths.";
            }
            model_reloading.store(false);
          });
    }

    // Skip rendering when hidden
    if (!window_visible) {
      continue;
    }

    // Render
    int fb_w, fb_h;
    glfwGetFramebufferSize(window, &fb_w, &fb_h);
    glViewport(0, 0, fb_w, fb_h);
    glClearColor(0.098f, 0.102f, 0.122f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);

    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();

    ImGui::SetNextWindowPos({0, 0});
    ImGui::SetNextWindowSize({(float)fb_w, (float)fb_h});
    ImGui::Begin("##main", nullptr,
                 ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
                     ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

    // Title bar
    {
      ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.345f, 0.525f, 0.976f, 1.f));
      ImGui::Text("\xf0\x9f\x94\x8d"); // magnifying-glass emoji
      ImGui::PopStyleColor();
      ImGui::SameLine();
      ImGui::Text("ohseeare");

      // Right-aligned settings / dismiss / quit
      {
        const char *settings_label = "\xe2\x9a\x99  Settings"; // ⚙ Settings
        const char *dismiss_label = "Dismiss";
        const char *quit_label = "Quit";
        float settings_w = ImGui::CalcTextSize(settings_label).x +
                           ImGui::GetStyle().FramePadding.x * 2.f;
        float dismiss_w = ImGui::CalcTextSize(dismiss_label).x +
                          ImGui::GetStyle().FramePadding.x * 2.f;
        float quit_w = ImGui::CalcTextSize(quit_label).x +
                       ImGui::GetStyle().FramePadding.x * 2.f;
        float spacing = ImGui::GetStyle().ItemSpacing.x;
        float right_x = ImGui::GetContentRegionAvail().x - settings_w -
                        dismiss_w - quit_w - spacing * 2.f;

        ImGui::SameLine(0, right_x);

        // Settings toggle button
        ImGui::PushStyleColor(ImGuiCol_Button,
                              settings_open
                                  ? ImVec4(0.265f, 0.440f, 0.890f, 1.f)
                                  : ImVec4(0.150f, 0.155f, 0.185f, 1.f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                              ImVec4(0.200f, 0.206f, 0.245f, 1.f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                              ImVec4(0.265f, 0.440f, 0.890f, 1.f));
        if (ImGui::Button(settings_label)) {
          settings_open = !settings_open;
          if (!settings_open) {
            // Save generation settings when panel closes
            cfg.temperature = settings_temperature;
            cfg.max_tokens = settings_max_tokens;
            cfg.auto_copy = settings_auto_copy;
          }
        }
        ImGui::PopStyleColor(3);

        ImGui::SameLine();

        ImGui::PushStyleColor(ImGuiCol_Button,
                              ImVec4(0.150f, 0.155f, 0.185f, 1.f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                              ImVec4(0.200f, 0.206f, 0.245f, 1.f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                              ImVec4(0.250f, 0.256f, 0.295f, 1.f));
        if (ImGui::Button(dismiss_label)) {
          window_hide_requested.store(true);
        }
        ImGui::PopStyleColor(3);

        ImGui::SameLine();

        // Quit button — danger red
        ImGui::PushStyleColor(ImGuiCol_Button,
                              ImVec4(0.55f, 0.15f, 0.15f, 1.f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                              ImVec4(0.70f, 0.20f, 0.20f, 1.f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                              ImVec4(0.45f, 0.10f, 0.10f, 1.f));
        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.f, 1.f, 1.f, 1.f));
        if (ImGui::Button(quit_label)) {
          app_quit_requested.store(true);
        }
        ImGui::PopStyleColor(4);
      }
    }

    ImGui::Spacing();
    ImGui::Separator();
    ImGui::Spacing();

    if (settings_open) {
      ImGui::PushStyleColor(ImGuiCol_ChildBg,
                            ImVec4(0.118f, 0.122f, 0.145f, 1.f));
      ImGui::BeginChild("##settings_panel", {-1, 0},
                        ImGuiChildFlags_AutoResizeY |
                            ImGuiChildFlags_FrameStyle);

      ImGui::TextDisabled("SETTINGS");
      ImGui::Spacing();

      bool reloading = model_reloading.load();
      if (reloading) {
        ImGui::BeginDisabled();
      }

      ImGui::TextDisabled("MODEL  \xe2\x80\x94  changes require reload");
      ImGui::Spacing();

      // Model path row
      ImGui::Text("Model (.gguf) ");
      ImGui::SameLine();
      char model_buf[512];
      snprintf(model_buf, sizeof(model_buf), "%s", settings_model_path.c_str());
      ImGui::SetNextItemWidth(-90);
      ImGui::InputText("##model_display", model_buf, sizeof(model_buf),
                       ImGuiInputTextFlags_ReadOnly);
      ImGui::SameLine();
      if (ImGui::Button("Browse##model", {80, 0})) {
        const char *filters[] = {"*.gguf"};
        const char *r = tinyfd_openFileDialog("Select Model File",
                                              settings_model_path.c_str(), 1,
                                              filters, "GGUF model files", 0);
        if (r)
          settings_model_path = r;
      }
      ImGui::Spacing();

      // mmproj path row
      ImGui::Text("mmproj (.gguf)");
      ImGui::SameLine();
      char mmproj_buf[512];
      snprintf(mmproj_buf, sizeof(mmproj_buf), "%s",
               settings_mmproj_path.c_str());
      ImGui::SetNextItemWidth(-90);
      ImGui::InputText("##mmproj_display", mmproj_buf, sizeof(mmproj_buf),
                       ImGuiInputTextFlags_ReadOnly);
      ImGui::SameLine();
      if (ImGui::Button("Browse##mmproj", {80, 0})) {
        const char *filters[] = {"*.gguf"};
        const char *r = tinyfd_openFileDialog("Select mmproj File",
                                              settings_mmproj_path.c_str(), 1,
                                              filters, "GGUF mmproj files", 0);
        if (r) {
          settings_mmproj_path = r;
        }
      }
      ImGui::Spacing();

      // CPU threads row
      ImGui::Text("CPU threads  ");
      ImGui::SameLine();
      ImGui::SetNextItemWidth(120);
      ImGui::DragInt("##n_threads", &settings_n_threads, 1, 1, 64);
      ImGui::SameLine();
      ImGui::TextDisabled("(1\xe2\x80\x93"
                          "64)");
      ImGui::Spacing();
      ImGui::Separator();
      ImGui::Spacing();

      // Reload button
      bool model_changed = (settings_model_path != model_path_str ||
                            settings_mmproj_path != mmproj_path_str ||
                            settings_n_threads != cfg.n_threads);

      if (!model_changed) {
        ImGui::BeginDisabled();
      }

      ImGui::PushStyleColor(ImGuiCol_Button,
                            ImVec4(0.345f, 0.525f, 0.976f, 1.f));
      ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                            ImVec4(0.435f, 0.600f, 1.000f, 1.f));
      ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                            ImVec4(0.265f, 0.440f, 0.890f, 1.f));

      if (ImGui::Button(reloading ? "Reloading..." : "Reload Model",
                        {120, 0})) {
        model_reload_requested.store(true);
      }
      ImGui::PopStyleColor(3);
      if (!model_changed) {
        ImGui::EndDisabled();
      }

      ImGui::SameLine();
      {
        std::lock_guard<std::mutex> lock(result_mutex);
        if (reloading) {
          ImGui::TextDisabled("Loading model, please wait...");
        } else if (!model_reload_status.empty()) {
          ImGui::TextDisabled("%s", model_reload_status.c_str());
        }
      }

      if (reloading) {
        ImGui::EndDisabled();
      }

      ImGui::Spacing();
      ImGui::Separator();
      ImGui::Spacing();

      // Generation section
      ImGui::TextDisabled("GENERATION  \xe2\x80\x94  takes effect immediately");
      ImGui::Spacing();

      // Prompt row
      ImGui::Text("Prompt       ");
      ImGui::SameLine();
      ImGui::SetNextItemWidth(-1);
      ImGui::InputText("##prompt", prompt, sizeof(prompt));
      ImGui::Spacing();

      // Temperature row
      ImGui::Text("Temperature  ");
      ImGui::SameLine();
      ImGui::SetNextItemWidth(-1);
      ImGui::SliderFloat("##temperature", &settings_temperature, 0.0f, 2.0f,
                         "%.2f");
      ImGui::Spacing();

      // Max tokens row
      ImGui::Text("Max tokens   ");
      ImGui::SameLine();
      ImGui::SetNextItemWidth(120);
      ImGui::DragInt("##max_tokens", &settings_max_tokens, 16, 64, 8192);
      ImGui::SameLine();
      ImGui::TextDisabled("(64\xe2\x80\x93"
                          "8192)");
      ImGui::Spacing();
      ImGui::Separator();
      ImGui::Spacing();

      // Auto-copy checkbox
      if (ImGui::Checkbox("Auto-copy result to clipboard",
                          &settings_auto_copy)) {
        cfg.auto_copy = settings_auto_copy;
        cfg.temperature = settings_temperature;
        cfg.max_tokens = settings_max_tokens;
      }

      ImGui::EndChild();
      ImGui::PopStyleColor();
      ImGui::Spacing();
      ImGui::Separator();
      ImGui::Spacing();
    }

    // Capture button
    bool running = ocr_running.load();
    {
      ImVec4 btn_col = ImVec4(0.345f, 0.525f, 0.976f, 1.f);
      ImVec4 btn_hov = ImVec4(0.435f, 0.600f, 1.000f, 1.f);
      ImVec4 btn_act = ImVec4(0.265f, 0.440f, 0.890f, 1.f);
      ImGui::PushStyleColor(ImGuiCol_Button, btn_col);
      ImGui::PushStyleColor(ImGuiCol_ButtonHovered, btn_hov);
      ImGui::PushStyleColor(ImGuiCol_ButtonActive, btn_act);
      ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1, 1, 1, 1));

      if (running) {
        ImGui::BeginDisabled();
      }
      if (ImGui::Button(
              running ? "  Processing\xe2\x80\xa6  "
                      : "  \xf0\x9f\x93\xb7  Capture Region  (Ctrl+Shift+S)  ",
              {-1, 36})) {
        capture_requested.store(true);
      }
      if (running) {
        ImGui::EndDisabled();
      }

      ImGui::PopStyleColor(4);
    }

    ImGui::Spacing();

    // Live timer while running
    if (timer_active) {
      double live_elapsed =
          std::chrono::duration<double>(Clock::now() - timer_start).count();
      char tbuf[64];
      format_elapsed(live_elapsed, tbuf, sizeof(tbuf));

      int dots = ((int)(live_elapsed * 2.0)) % 4;
      const char *dot_str[] = {"", ".", "..", "..."};

      ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.976f, 0.729f, 0.345f, 1.f));
      ImGui::Text("\xe2\x8f\xb1  %s elapsed%s", tbuf, dot_str[dots]);
      ImGui::PopStyleColor();
      ImGui::Spacing();
    }

    // Result section
    ImGui::TextDisabled("RESULT");
    ImGui::Spacing();

    ImVec2 avail = ImGui::GetContentRegionAvail();
    float status_bar_h = ImGui::GetFrameHeightWithSpacing() + 8.f;
    float text_h = avail.y - status_bar_h;

    if (text_h < 60.f) {
      text_h = 60.f;
    }

    // Copy result into a stable local buffer under the lock
    std::vector<char> result_buf;
    {
      std::lock_guard<std::mutex> lock(result_mutex);
      result_buf.assign(result.begin(), result.end());
      result_buf.push_back('\0');
    }

    ImGui::PushStyleColor(ImGuiCol_FrameBg,
                          ImVec4(0.118f, 0.122f, 0.145f, 1.f));
    if (font_mono) {
      ImGui::PushFont(font_mono);
    }

    ImGui::InputTextMultiline("##result", result_buf.data(), result_buf.size(),
                              {-1, text_h}, ImGuiInputTextFlags_ReadOnly);
    if (font_mono) {
      ImGui::PopFont();
    }

    ImGui::PopStyleColor();

    // Status bar
    ImGui::Spacing();
    {
      // Copy button
      ImGui::PushStyleColor(ImGuiCol_Button,
                            ImVec4(0.150f, 0.155f, 0.185f, 1.f));
      ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                            ImVec4(0.200f, 0.206f, 0.245f, 1.f));
      ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                            ImVec4(0.265f, 0.440f, 0.890f, 1.f));
      if (ImGui::Button("\xf0\x9f\x93\x8b  Copy", {80, 0})) {
        std::lock_guard<std::mutex> lock(result_mutex);
        ImGui::SetClipboardText(result.c_str());
        status = "Copied to clipboard!";
      }
      ImGui::PopStyleColor(3);

      ImGui::SameLine();

      // Rerun button - re-run OCR on the last captured image
      ImGui::PushStyleColor(ImGuiCol_Button,
                            ImVec4(0.150f, 0.155f, 0.185f, 1.f));
      ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                            ImVec4(0.200f, 0.206f, 0.245f, 1.f));
      ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                            ImVec4(0.265f, 0.440f, 0.890f, 1.f));
      bool can_rerun = !running && last_cap != nullptr;

      if (!can_rerun) {
        ImGui::BeginDisabled();
      }
      if (ImGui::Button("\xe2\x86\xba  Rerun", {80, 0})) {
        rerun_requested.store(true);
      }
      if (!can_rerun) {
        ImGui::EndDisabled();
      }

      ImGui::PopStyleColor(3);

      ImGui::SameLine();
      ImGui::TextDisabled(" \xc2\xb7 ");
      ImGui::SameLine();

      // Status text (colour-coded)
      {
        std::lock_guard<std::mutex> lock(result_mutex);
        if (running) {
          ImGui::PushStyleColor(ImGuiCol_Text,
                                ImVec4(0.976f, 0.729f, 0.345f, 1.f));
          ImGui::TextUnformatted(status.c_str());
          ImGui::PopStyleColor();
        } else if (status.find("Done") != std::string::npos) {
          ImGui::PushStyleColor(ImGuiCol_Text,
                                ImVec4(0.400f, 0.850f, 0.500f, 1.f));
          ImGui::TextUnformatted(status.c_str());
          ImGui::PopStyleColor();
        } else {
          ImGui::TextDisabled("%s", status.c_str());
        }
      }

      // Escape hint on the far right
      ImGui::SameLine(ImGui::GetContentRegionMax().x -
                      ImGui::CalcTextSize("Esc to dismiss").x);
      ImGui::TextDisabled("Esc to dismiss");
    }

    ImGui::End();

    ImGui::Render();
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
    glfwSwapBuffers(window);
  }

  // Cleanup
  if (ocr_thread.joinable()) {
    ocr_thread.join();
  }
  if (reload_thread.joinable()) {
    reload_thread.join();
  }

  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImGui::DestroyContext();
  glfwDestroyWindow(window);
  glfwTerminate();
  return 0;
}
