//! Color themes for the Ratatui UI.

use ratatui::style::{Color, Modifier, Style};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThemeId {
    #[default]
    Dark,
    Ocean,
    Ember,
    Mono,
}

impl ThemeId {
    pub fn name(self) -> &'static str {
        match self {
            ThemeId::Dark => "dark",
            ThemeId::Ocean => "ocean",
            ThemeId::Ember => "ember",
            ThemeId::Mono => "mono",
        }
    }

    pub fn next(self) -> Self {
        match self {
            ThemeId::Dark => ThemeId::Ocean,
            ThemeId::Ocean => ThemeId::Ember,
            ThemeId::Ember => ThemeId::Mono,
            ThemeId::Mono => ThemeId::Dark,
        }
    }

    pub fn palette(self) -> Theme {
        match self {
            ThemeId::Dark => Theme {
                user: Color::Cyan,
                assistant: Color::Green,
                thinking: Color::DarkGray,
                tool: Color::Yellow,
                notice_info: Color::DarkGray,
                notice_warn: Color::Magenta,
                notice_err: Color::Red,
                header: Color::DarkGray,
                footer: Color::Gray,
                footer_busy: Color::Yellow,
                input_border: Color::Cyan,
                input_border_busy: Color::DarkGray,
                code: Color::LightBlue,
                bold: Color::White,
                md_heading: Color::LightCyan,
                dim: Color::DarkGray,
            },
            ThemeId::Ocean => Theme {
                user: Color::LightCyan,
                assistant: Color::LightGreen,
                thinking: Color::Rgb(80, 100, 120),
                tool: Color::LightYellow,
                notice_info: Color::Rgb(100, 140, 160),
                notice_warn: Color::Rgb(220, 180, 80),
                notice_err: Color::LightRed,
                header: Color::Rgb(90, 120, 150),
                footer: Color::Rgb(140, 180, 200),
                footer_busy: Color::LightYellow,
                input_border: Color::LightCyan,
                input_border_busy: Color::Rgb(60, 90, 110),
                code: Color::Rgb(120, 200, 255),
                bold: Color::White,
                md_heading: Color::Rgb(100, 220, 255),
                dim: Color::Rgb(70, 100, 120),
            },
            ThemeId::Ember => Theme {
                user: Color::Rgb(255, 180, 100),
                assistant: Color::Rgb(255, 200, 140),
                thinking: Color::Rgb(120, 80, 60),
                tool: Color::Rgb(255, 160, 60),
                notice_info: Color::Rgb(140, 110, 90),
                notice_warn: Color::Rgb(255, 140, 60),
                notice_err: Color::Rgb(255, 80, 60),
                header: Color::Rgb(120, 90, 70),
                footer: Color::Rgb(200, 160, 120),
                footer_busy: Color::Rgb(255, 180, 80),
                input_border: Color::Rgb(255, 160, 80),
                input_border_busy: Color::Rgb(100, 70, 50),
                code: Color::Rgb(255, 200, 120),
                bold: Color::Rgb(255, 230, 200),
                md_heading: Color::Rgb(255, 160, 80),
                dim: Color::Rgb(100, 70, 50),
            },
            ThemeId::Mono => Theme {
                user: Color::White,
                assistant: Color::Gray,
                thinking: Color::DarkGray,
                tool: Color::White,
                notice_info: Color::DarkGray,
                notice_warn: Color::Gray,
                notice_err: Color::White,
                header: Color::DarkGray,
                footer: Color::Gray,
                footer_busy: Color::White,
                input_border: Color::White,
                input_border_busy: Color::DarkGray,
                code: Color::White,
                bold: Color::White,
                md_heading: Color::White,
                dim: Color::DarkGray,
            },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Theme {
    pub user: Color,
    pub assistant: Color,
    pub thinking: Color,
    pub tool: Color,
    pub notice_info: Color,
    pub notice_warn: Color,
    pub notice_err: Color,
    pub header: Color,
    pub footer: Color,
    pub footer_busy: Color,
    pub input_border: Color,
    pub input_border_busy: Color,
    pub code: Color,
    pub bold: Color,
    pub md_heading: Color,
    pub dim: Color,
}

impl Theme {
    pub fn role(self, name: &str) -> Style {
        Style::default()
            .fg(match name {
                "you" | "user" => self.user,
                "assistant" => self.assistant,
                "thinking" => self.thinking,
                _ => self.dim,
            })
            .add_modifier(Modifier::BOLD)
    }

    pub fn notice(self, severity: &str) -> Style {
        Style::default().fg(match severity {
            "error" => self.notice_err,
            "warning" => self.notice_warn,
            _ => self.notice_info,
        })
    }
}
