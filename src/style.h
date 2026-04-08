#pragma once
#include "imgui.h"

static void apply_custom_style() {
  ImGuiStyle &s = ImGui::GetStyle();

  // Geometry
  s.WindowPadding = ImVec2(20, 16);
  s.FramePadding = ImVec2(15, 6);
  s.ItemSpacing = ImVec2(10, 8);
  s.ItemInnerSpacing = ImVec2(8, 4);
  s.ScrollbarSize = 12.f;
  s.GrabMinSize = 10.f;

  // Rounding
  s.WindowRounding = 5.f;
  s.FrameRounding = 6.f;
  s.GrabRounding = 4.f;
  s.ScrollbarRounding = 6.f;
  s.TabRounding = 6.f;
  s.ChildRounding = 6.f;
  s.PopupRounding = 6.f;

  // Borders
  s.WindowBorderSize = 3.f;
  s.FrameBorderSize = 0.f;
  s.PopupBorderSize = 1.f;

  // Colours - slate-blue dark theme
  ImVec4 *c = s.Colors;

  const ImVec4 bg = ImVec4(0.098f, 0.102f, 0.122f, 1.00f);
  const ImVec4 bg_child = ImVec4(0.118f, 0.122f, 0.145f, 1.00f);
  const ImVec4 surface = ImVec4(0.150f, 0.155f, 0.185f, 1.00f);
  const ImVec4 surface_hi = ImVec4(0.190f, 0.195f, 0.230f, 1.00f);
  const ImVec4 accent = ImVec4(0.345f, 0.525f, 0.976f, 1.00f);
  const ImVec4 accent_hov = ImVec4(0.435f, 0.600f, 1.000f, 1.00f);
  const ImVec4 accent_act = ImVec4(0.265f, 0.440f, 0.890f, 1.00f);
  const ImVec4 text = ImVec4(0.900f, 0.910f, 0.940f, 1.00f);
  const ImVec4 text_dim = ImVec4(0.500f, 0.520f, 0.580f, 1.00f);
  const ImVec4 border = ImVec4(0.220f, 0.225f, 0.265f, 1.00f);

  c[ImGuiCol_WindowBg] = bg;
  c[ImGuiCol_ChildBg] = bg_child;
  c[ImGuiCol_PopupBg] = bg_child;
  c[ImGuiCol_Border] = border;

  c[ImGuiCol_Text] = text;
  c[ImGuiCol_TextDisabled] = text_dim;

  c[ImGuiCol_FrameBg] = surface;
  c[ImGuiCol_FrameBgHovered] = surface_hi;
  c[ImGuiCol_FrameBgActive] = surface_hi;

  c[ImGuiCol_TitleBg] = bg;
  c[ImGuiCol_TitleBgActive] = bg;
  c[ImGuiCol_TitleBgCollapsed] = bg;
  c[ImGuiCol_MenuBarBg] = bg;

  c[ImGuiCol_ScrollbarBg] = bg_child;
  c[ImGuiCol_ScrollbarGrab] = surface_hi;
  c[ImGuiCol_ScrollbarGrabHovered] = ImVec4(0.25f, 0.26f, 0.30f, 1.f);
  c[ImGuiCol_ScrollbarGrabActive] = ImVec4(0.30f, 0.31f, 0.36f, 1.f);

  c[ImGuiCol_Button] = surface;
  c[ImGuiCol_ButtonHovered] = accent_hov;
  c[ImGuiCol_ButtonActive] = accent_act;

  c[ImGuiCol_Header] = surface;
  c[ImGuiCol_HeaderHovered] = surface_hi;
  c[ImGuiCol_HeaderActive] = surface_hi;

  c[ImGuiCol_Separator] = border;
  c[ImGuiCol_SeparatorHovered] = accent;
  c[ImGuiCol_SeparatorActive] = accent;

  c[ImGuiCol_CheckMark] = accent;
  c[ImGuiCol_SliderGrab] = accent;
  c[ImGuiCol_SliderGrabActive] = accent_act;

  c[ImGuiCol_ResizeGrip] = surface;
  c[ImGuiCol_ResizeGripHovered] = accent_hov;
  c[ImGuiCol_ResizeGripActive] = accent_act;

  c[ImGuiCol_Tab] = surface;
  c[ImGuiCol_TabHovered] = accent_hov;

  c[ImGuiCol_TextSelectedBg] = ImVec4(accent.x, accent.y, accent.z, 0.35f);
  c[ImGuiCol_NavHighlight] = accent;
}

