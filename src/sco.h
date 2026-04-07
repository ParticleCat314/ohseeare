#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#if defined(__linux__)
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#elif defined(_WIN32)
#include <windows.h>
#endif

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>


struct ScoImage {
    std::vector<uint8_t> rgba;  // row-major, top-to-bottom, 4 bytes/pixel (R G B A)
    int width  = 0;
    int height = 0;
};

#if defined(__linux__)
static ScoImage grab_desktop_impl() {
    ScoImage img;

    Display *dpy = XOpenDisplay(nullptr);
    if (!dpy)
        return img;

    Window root = DefaultRootWindow(dpy);
    XWindowAttributes attrs;
    XGetWindowAttributes(dpy, root, &attrs);

    img.width  = attrs.width;
    img.height = attrs.height;

    XImage *ximg = XGetImage(dpy, root, 0, 0, (unsigned)img.width,
                             (unsigned)img.height, AllPlanes, ZPixmap);
    if (!ximg) {
        XCloseDisplay(dpy);
        img.width = img.height = 0;
        return img;
    }

    img.rgba.resize((size_t)img.width * img.height * 4);

    if (ximg->bits_per_pixel == 32) {
        const bool le = (ximg->byte_order == LSBFirst);
        const int ri  = le ? 2 : 1;
        const int gi  = le ? 1 : 2;
        const int bi  = le ? 0 : 3;

        for (int y = 0; y < img.height; y++) {
            const uint8_t *src =
                (const uint8_t *)ximg->data + (size_t)y * ximg->bytes_per_line;
            uint8_t *dst = img.rgba.data() + (size_t)y * img.width * 4;
            for (int x = 0; x < img.width; x++) {
                dst[0] = src[ri];
                dst[1] = src[gi];
                dst[2] = src[bi];
                dst[3] = 0xFF;
                src += 4;
                dst += 4;
            }
        }
    } else {
        for (int y = 0; y < img.height; y++) {
            for (int x = 0; x < img.width; x++) {
                unsigned long p = XGetPixel(ximg, x, y);
                size_t idx = ((size_t)y * img.width + x) * 4;
                img.rgba[idx + 0] = (uint8_t)((p >> 16) & 0xFF);
                img.rgba[idx + 1] = (uint8_t)((p >> 8)  & 0xFF);
                img.rgba[idx + 2] = (uint8_t)((p >> 0)  & 0xFF);
                img.rgba[idx + 3] = 0xFF;
            }
        }
    }

    XDestroyImage(ximg);
    XCloseDisplay(dpy);
    return img;
}

#elif defined(_WIN32)
static ScoImage grab_desktop_impl() {
    ScoImage img;

    img.width  = GetSystemMetrics(SM_CXVIRTUALSCREEN);
    img.height = GetSystemMetrics(SM_CYVIRTUALSCREEN);
    int vx = GetSystemMetrics(SM_XVIRTUALSCREEN);
    int vy = GetSystemMetrics(SM_YVIRTUALSCREEN);

    if (img.width <= 0 || img.height <= 0)
        return img;

    HDC screenDC = GetDC(nullptr);
    HDC memDC    = CreateCompatibleDC(screenDC);
    HBITMAP hBmp = CreateCompatibleBitmap(screenDC, img.width, img.height);
    HGDIOBJ old  = SelectObject(memDC, hBmp);

    BitBlt(memDC, 0, 0, img.width, img.height, screenDC, vx, vy, SRCCOPY);

    BITMAPINFOHEADER bi = {};
    bi.biSize        = sizeof(bi);
    bi.biWidth       = img.width;
    bi.biHeight      = -img.height;
    bi.biPlanes      = 1;
    bi.biBitCount    = 32;
    bi.biCompression = BI_RGB;

    img.rgba.resize((size_t)img.width * img.height * 4);
    GetDIBits(memDC, hBmp, 0, img.height, img.rgba.data(),
              (BITMAPINFO *)&bi, DIB_RGB_COLORS);

    for (size_t i = 0; i < (size_t)img.width * img.height; i++) {
        std::swap(img.rgba[i * 4 + 0], img.rgba[i * 4 + 2]);
        img.rgba[i * 4 + 3] = 0xFF;
    }

    SelectObject(memDC, old);
    DeleteObject(hBmp);
    DeleteDC(memDC);
    ReleaseDC(nullptr, screenDC);
    return img;
}

#else
static ScoImage grab_desktop_impl() { return {}; }
#endif

ScoImage sco_grab_desktop() {
    return grab_desktop_impl();
}

struct OVtx {
    float x, y;
    float u, v;
    float r, g, b, a;
};

static void push_quad(std::vector<OVtx> &verts,
                      float x0, float y0, float x1, float y1,
                      float u0, float v0, float u1, float v1,
                      float r, float g, float b, float a) {
    verts.push_back({x0, y0, u0, v0, r, g, b, a});
    verts.push_back({x1, y0, u1, v0, r, g, b, a});
    verts.push_back({x1, y1, u1, v1, r, g, b, a});
    verts.push_back({x0, y0, u0, v0, r, g, b, a});
    verts.push_back({x1, y1, u1, v1, r, g, b, a});
    verts.push_back({x0, y1, u0, v1, r, g, b, a});
}

static void push_line(std::vector<OVtx> &verts,
                      float x0, float y0, float x1, float y1,
                      float r, float g, float b, float a) {
    verts.push_back({x0, y0, 0, 0, r, g, b, a});
    verts.push_back({x1, y1, 0, 0, r, g, b, a});
}

static const char *overlay_vert_src = R"(
#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec2 aUV;
layout(location = 2) in vec4 aColor;
uniform vec2 uScreen;
out vec2 vUV;
out vec4 vColor;
void main() {
    vec2 ndc = aPos / uScreen * 2.0 - 1.0;
    ndc.y = -ndc.y;
    gl_Position = vec4(ndc, 0.0, 1.0);
    vUV = aUV;
    vColor = aColor;
}
)";

static const char *overlay_frag_src = R"(
#version 330 core
in vec2 vUV;
in vec4 vColor;
uniform sampler2D uTex;
uniform int uMode; // 0 = color only, 1 = texture * color
out vec4 fragColor;
void main() {
    if (uMode == 1)
        fragColor = texture(uTex, vUV) * vColor;
    else
        fragColor = vColor;
}
)";

static GLuint compile_shader(GLenum type, const char *src) {
    GLuint s = glCreateShader(type);
    glShaderSource(s, 1, &src, nullptr);
    glCompileShader(s);
    GLint ok = 0;
    glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetShaderInfoLog(s, sizeof(log), nullptr, log);
        fprintf(stderr, "Shader compile error: %s\n", log);
        glDeleteShader(s);
        return 0;
    }
    return s;
}

static GLuint create_program(const char *vs_src, const char *fs_src) {
    GLuint vs = compile_shader(GL_VERTEX_SHADER, vs_src);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fs_src);
    if (!vs || !fs) {
        if (vs) glDeleteShader(vs);
        if (fs) glDeleteShader(fs);
        return 0;
    }
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);

    GLint ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
        fprintf(stderr, "Program link error: %s\n", log);
        glDeleteProgram(prog);
        prog = 0;
    }
    glDeleteShader(vs);
    glDeleteShader(fs);
    return prog;
}

ScoImage sco_capture_region() {
    // 1. Grab a screenshot of the desktop
    ScoImage shot = grab_desktop_impl();
    if (shot.rgba.empty()) {
        return {};
    }
    // 2. Compute overlay bounds from GLFW monitors
    int mon_count = 0;
    GLFWmonitor **mons = glfwGetMonitors(&mon_count);
    if (!mons || mon_count == 0) {
        return {};
    }

    int vx0 = 0, vy0 = 0, vx1 = 0, vy1 = 0;
    for (int i = 0; i < mon_count; i++) {
        int mx, my;
        glfwGetMonitorPos(mons[i], &mx, &my);
        const GLFWvidmode *mode = glfwGetVideoMode(mons[i]);
        if (!mode)
            continue;
        if (i == 0) {
            vx0 = mx;
            vy0 = my;
            vx1 = mx + mode->width;
            vy1 = my + mode->height;
        } else {
            vx0 = std::min(vx0, mx);
            vy0 = std::min(vy0, my);
            vx1 = std::max(vx1, mx + mode->width);
            vy1 = std::max(vy1, my + mode->height);
        }
    }
    int vw = vx1 - vx0;
    int vh = vy1 - vy0;
    if (vw <= 0 || vh <= 0) {
        printf("Invalid virtual desktop size\n");
        return {};
    }

    // 3. Save caller's GL context
    GLFWwindow *prev_glfw = glfwGetCurrentContext();

    // 4. Create overlay window
    glfwWindowHint(GLFW_DECORATED,      GLFW_FALSE);
    glfwWindowHint(GLFW_FLOATING,       GLFW_TRUE);
    glfwWindowHint(GLFW_RESIZABLE,      GLFW_FALSE);
    glfwWindowHint(GLFW_FOCUSED,        GLFW_TRUE);
    glfwWindowHint(GLFW_AUTO_ICONIFY,   GLFW_FALSE);
    glfwWindowHint(GLFW_FOCUS_ON_SHOW,  GLFW_TRUE);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow *overlay = glfwCreateWindow(vw, vh, "Capture", nullptr, nullptr);
    if (!overlay) {
        glfwDefaultWindowHints();
        if (prev_glfw)
            glfwMakeContextCurrent(prev_glfw);
        printf("Failed to create GLFW window\n");
        return {};
    }
    glfwSetWindowPos(overlay, vx0, vy0);
    glfwMakeContextCurrent(overlay);
    gladLoadGL();
    glfwSwapInterval(1);

    GLFWcursor *crosshair = glfwCreateStandardCursor(GLFW_CROSSHAIR_CURSOR);
    glfwSetCursor(overlay, crosshair);

    // 5. Set up shader & VAO/VBO
    GLuint prog = create_program(overlay_vert_src, overlay_frag_src);
    if (!prog) {
        glfwDestroyCursor(crosshair);
        glfwDestroyWindow(overlay);
        glfwDefaultWindowHints();
        if (prev_glfw)
            glfwMakeContextCurrent(prev_glfw);
        return {};
    }

    GLint loc_screen = glGetUniformLocation(prog, "uScreen");
    GLint loc_mode   = glGetUniformLocation(prog, "uMode");
    GLint loc_tex    = glGetUniformLocation(prog, "uTex");

    GLuint vao, vbo;
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    // position
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(OVtx),
                          (void *)offsetof(OVtx, x));
    // uv
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(OVtx),
                          (void *)offsetof(OVtx, u));
    // color
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(2, 4, GL_FLOAT, GL_FALSE, sizeof(OVtx),
                          (void *)offsetof(OVtx, r));

    // 6. Upload screenshot as GL texture
    GLuint tex = 0;
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, shot.width, shot.height, 0, GL_RGBA,
                 GL_UNSIGNED_BYTE, shot.rgba.data());

    // 7. Selection loop
    bool   confirmed = false;
    bool   dragging  = false;
    float  sel_sx = 0, sel_sy = 0;
    float  sel_ex = 0, sel_ey = 0;
    bool   prev_lmb = false;
    std::vector<OVtx> verts;

    while (!glfwWindowShouldClose(overlay)) {
        printf("\rSelection in progress... Press Escape or right-click to cancel. ");
        glfwPollEvents();

        if (glfwGetKey(overlay, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            break;
        if (glfwGetMouseButton(overlay, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS)
            break;

        int win_w, win_h;
        glfwGetWindowSize(overlay, &win_w, &win_h);
        int fb_w, fb_h;
        glfwGetFramebufferSize(overlay, &fb_w, &fb_h);

        double dmx, dmy;
        glfwGetCursorPos(overlay, &dmx, &dmy);
        float mouse_x = (float)dmx;
        float mouse_y = (float)dmy;
        float disp_w  = (float)win_w;
        float disp_h  = (float)win_h;

        // Mouse state tracking
        bool curr_lmb = glfwGetMouseButton(overlay, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
        bool clicked  = curr_lmb && !prev_lmb;
        bool released = !curr_lmb && prev_lmb;
        prev_lmb = curr_lmb;

        if (!dragging && clicked) {
            dragging = true;
            sel_sx = mouse_x;
            sel_sy = mouse_y;
            sel_ex = mouse_x;
            sel_ey = mouse_y;
        }
        if (dragging) {
            sel_ex = mouse_x;
            sel_ey = mouse_y;
            if (released) {
                dragging = false;
                float smin_x = std::min(sel_sx, sel_ex);
                float smin_y = std::min(sel_sy, sel_ey);
                float smax_x = std::max(sel_sx, sel_ex);
                float smax_y = std::max(sel_sy, sel_ey);
                if ((smax_x - smin_x) >= 5.0f && (smax_y - smin_y) >= 5.0f) {
                    confirmed = true;
                    break;
                }
            }
        }

        // Normalised selection rectangle
        float sel_min_x = std::max(std::min(sel_sx, sel_ex), 0.0f);
        float sel_min_y = std::max(std::min(sel_sy, sel_ey), 0.0f);
        float sel_max_x = std::min(std::max(sel_sx, sel_ex), disp_w);
        float sel_max_y = std::min(std::max(sel_sy, sel_ey), disp_h);
        bool has_sel = (sel_max_x - sel_min_x) > 1.0f &&
                       (sel_max_y - sel_min_y) > 1.0f;

        // --- Render ---
        glViewport(0, 0, fb_w, fb_h);
        glClearColor(0, 0, 0, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

        glUseProgram(prog);
        glUniform2f(loc_screen, disp_w, disp_h);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, tex);
        glUniform1i(loc_tex, 0);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);

        // Background screenshot (textured)
        verts.clear();
        push_quad(verts, 0, 0, disp_w, disp_h, 0, 0, 1, 1, 1, 1, 1, 1);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(verts.size() * sizeof(OVtx)),
                     verts.data(), GL_STREAM_DRAW);
        glUniform1i(loc_mode, 1);
        glDrawArrays(GL_TRIANGLES, 0, (GLsizei)verts.size());

        // Dim overlay (semi-transparent black)
        verts.clear();
        push_quad(verts, 0, 0, disp_w, disp_h, 0, 0, 0, 0,
                  0, 0, 0, 120.0f / 255.0f);
        glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(verts.size() * sizeof(OVtx)),
                     verts.data(), GL_STREAM_DRAW);
        glUniform1i(loc_mode, 0);
        glDrawArrays(GL_TRIANGLES, 0, (GLsizei)verts.size());

        if (has_sel) {
            // Un-dimmed selection region (textured)
            float u0 = sel_min_x / disp_w;
            float v0 = sel_min_y / disp_h;
            float u1 = sel_max_x / disp_w;
            float v1 = sel_max_y / disp_h;
            verts.clear();
            push_quad(verts, sel_min_x, sel_min_y, sel_max_x, sel_max_y,
                      u0, v0, u1, v1, 1, 1, 1, 1);
            glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(verts.size() * sizeof(OVtx)),
                         verts.data(), GL_STREAM_DRAW);
            glUniform1i(loc_mode, 1);
            glDrawArrays(GL_TRIANGLES, 0, (GLsizei)verts.size());

            // Selection border
            float br = 100.0f / 255.0f, bg = 150.0f / 255.0f, bb = 1.0f;
            verts.clear();
            push_line(verts, sel_min_x, sel_min_y, sel_max_x, sel_min_y, br, bg, bb, 1);
            push_line(verts, sel_max_x, sel_min_y, sel_max_x, sel_max_y, br, bg, bb, 1);
            push_line(verts, sel_max_x, sel_max_y, sel_min_x, sel_max_y, br, bg, bb, 1);
            push_line(verts, sel_min_x, sel_max_y, sel_min_x, sel_min_y, br, bg, bb, 1);
            glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(verts.size() * sizeof(OVtx)),
                         verts.data(), GL_STREAM_DRAW);
            glUniform1i(loc_mode, 0);
            glLineWidth(2.0f);
            glDrawArrays(GL_LINES, 0, (GLsizei)verts.size());
        }

        // Crosshair guide lines
        if (!has_sel || dragging) {
            float ca = 60.0f / 255.0f;
            verts.clear();
            push_line(verts, mouse_x, 0, mouse_x, disp_h, 1, 1, 1, ca);
            push_line(verts, 0, mouse_y, disp_w, mouse_y, 1, 1, 1, ca);
            glBufferData(GL_ARRAY_BUFFER, (GLsizeiptr)(verts.size() * sizeof(OVtx)),
                         verts.data(), GL_STREAM_DRAW);
            glUniform1i(loc_mode, 0);
            glLineWidth(1.0f);
            glDrawArrays(GL_LINES, 0, (GLsizei)verts.size());
        }

        glDisable(GL_BLEND);
        glfwSwapBuffers(overlay);
    }

    // 8. Crop and return the selected region
    ScoImage result;
    if (confirmed) {
        float smin_x = std::max(std::min(sel_sx, sel_ex), 0.0f);
        float smin_y = std::max(std::min(sel_sy, sel_ey), 0.0f);
        float smax_x = std::min(std::max(sel_sx, sel_ex), (float)vw);
        float smax_y = std::min(std::max(sel_sy, sel_ey), (float)vh);

        float sx = (float)shot.width  / (float)vw;
        float sy = (float)shot.height / (float)vh;

        int cx = (int)(smin_x * sx);
        int cy = (int)(smin_y * sy);
        int cw = (int)((smax_x - smin_x) * sx);
        int ch = (int)((smax_y - smin_y) * sy);

        cx = std::max(0, std::min(cx, shot.width  - 1));
        cy = std::max(0, std::min(cy, shot.height - 1));
        cw = std::min(cw, shot.width  - cx);
        ch = std::min(ch, shot.height - cy);

        if (cw > 0 && ch > 0) {
            result.width  = cw;
            result.height = ch;
            result.rgba.resize((size_t)cw * ch * 4);
            for (int y = 0; y < ch; y++) {
                const uint8_t *src =
                    shot.rgba.data() + ((size_t)(cy + y) * shot.width + cx) * 4;
                uint8_t *dst = result.rgba.data() + (size_t)y * cw * 4;
                std::memcpy(dst, src, (size_t)cw * 4);
            }
        }
    }

    // 9. Cleanup
    glDeleteTextures(1, &tex);
    glDeleteBuffers(1, &vbo);
    glDeleteVertexArrays(1, &vao);
    glDeleteProgram(prog);

    glfwDestroyCursor(crosshair);
    glfwDestroyWindow(overlay);

    if (prev_glfw)
        glfwMakeContextCurrent(prev_glfw);
    glfwDefaultWindowHints();

    return result;
}
