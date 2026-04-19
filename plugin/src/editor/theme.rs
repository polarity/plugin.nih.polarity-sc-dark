// Spectral Compressor: an FFT based compressor
// Copyright (C) 2021-2024 Robbert van der Helm
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

//! NanoVG colors used by the analyzer's custom canvas drawing. Everything
//! reachable through vizia styling (backgrounds, text, borders, ...) lives in
//! `theme.css` instead so it can be tweaked with the `hot-reload` feature.

use nih_plug_vizia::vizia::vg;

// Analyzer background and spectrum.
pub const ANALYZER_BACKGROUND: vg::Color = vg::Color::rgbaf(0.137, 0.137, 0.137, 1.0);
pub const ANALYZER_SPECTRUM_BARS: vg::Color = vg::Color::rgbaf(0.75, 0.75, 0.75, 0.85);
pub const ANALYZER_SPECTRUM_MESH_LIGHT: vg::Color = vg::Color::rgbaf(0.82, 0.82, 0.82, 0.90);
pub const ANALYZER_SPECTRUM_MESH_DARK: vg::Color = vg::Color::rgbaf(0.75, 0.75, 0.75, 0.85);

// Gain reduction overlays: yellow for downwards, blue for upwards.
pub const ANALYZER_GR_DOWNWARDS: vg::Color = vg::Color::rgbaf(0.902, 0.698, 0.227, 0.65);
pub const ANALYZER_GR_UPWARDS: vg::Color = vg::Color::rgbaf(0.416, 0.784, 0.847, 0.65);

// Threshold curves matching the overlay colors above.
pub const ANALYZER_THRESHOLD_DOWNWARDS: vg::Color = vg::Color::rgbaf(0.902, 0.698, 0.227, 0.9);
pub const ANALYZER_THRESHOLD_UPWARDS: vg::Color = vg::Color::rgbaf(0.416, 0.784, 0.847, 0.9);
