import pygame
import os
from .config import Colors


def get_font(size, bold=False):
    """Return a font that supports Latin; tries common CJK-capable fonts if available."""
    local_fonts = ['SourceHanSansCN-Regular.otf', 'font.ttf', 'SimHei.ttf']
    for font_file in local_fonts:
        if os.path.exists(font_file):
            try:
                return pygame.font.Font(font_file, size)
            except Exception:
                continue

    font_names = [
        'sourcehansanscn', 'notosanssc', 'microsoftyaheiui', 'microsoftyahei',
        'pingfangsc', 'heiti', 'simhei', 'arial', 'helvetica', 'arialunicodems'
    ]
    for name in font_names:
        try:
            path = pygame.font.match_font(name)
            if path:
                return pygame.font.Font(path, size)
        except Exception:
            continue
    return pygame.font.SysFont('arial', size, bold=bold)


def draw_panel(surface, rect, alpha=230):
    """Draw a semi-transparent rounded panel with a drop shadow."""
    shadow_rect = pygame.Rect(rect[0]+4, rect[1]+4, rect[2], rect[3])
    s = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
    pygame.draw.rect(s, (0, 0, 0, 100), s.get_rect(), border_radius=15)
    surface.blit(s, shadow_rect.topleft)

    s_main = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
    bg_color = list(Colors.UI_PANEL_BG)
    bg_color[3] = alpha
    pygame.draw.rect(s_main, tuple(bg_color), s_main.get_rect(), border_radius=15)
    pygame.draw.rect(s_main, Colors.UI_BORDER, s_main.get_rect(), 2, border_radius=15)
    surface.blit(s_main, rect[:2])


class Button:
    def __init__(self, x, y, w, h, text, callback):
        self.rect = pygame.Rect(x, y, w, h)
        self.text = text
        self.callback = callback
        self.is_hovered = False
        self.animation_offset = 0

    def update(self):
        target = -4 if self.is_hovered else 0
        self.animation_offset += (target - self.animation_offset) * 0.2

    def draw(self, surface, font):
        draw_rect = self.rect.copy()
        draw_rect.y += self.animation_offset

        shadow_rect = draw_rect.copy()
        shadow_rect.move_ip(2, 4)
        pygame.draw.rect(surface, (0,0,0,80), shadow_rect, border_radius=12)

        color = Colors.BTN_HOVER if self.is_hovered else Colors.BTN_NORMAL
        pygame.draw.rect(surface, color, draw_rect, border_radius=12)
        pygame.draw.rect(surface, (255,255,255, 100), draw_rect, 2, border_radius=12)

        text_surf = font.render(self.text, True, Colors.BTN_TEXT)
        text_rect = text_surf.get_rect(center=draw_rect.center)
        surface.blit(text_surf, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.is_hovered:
                if self.callback:
                    self.callback()
