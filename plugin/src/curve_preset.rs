use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use directories::ProjectDirs;
use serde::{Deserialize, Serialize};

use crate::curve::{
    CurvePoint, MAX_THRESHOLD_CURVE_POINTS, THRESHOLD_CURVE_MAX_FREQUENCY_HZ,
    THRESHOLD_CURVE_MIN_FREQUENCY_HZ, THRESHOLD_CURVE_POINT_OFFSET_LIMIT_DB,
};

const PRESET_FORMAT_VERSION: u32 = 1;
const USER_PRESET_FOLDER: &str = "curve-presets";

const THRESHOLD_MIN_DB: f32 = -100.0;
const THRESHOLD_MAX_DB: f32 = 20.0;
const CURVE_SLOPE_MIN: f32 = -36.0;
const CURVE_SLOPE_MAX: f32 = 36.0;
const CURVE_CURVE_MIN: f32 = -24.0;
const CURVE_CURVE_MAX: f32 = 24.0;

#[derive(Debug, Clone)]
pub(crate) struct CurvePreset {
    pub(crate) name: String,
    pub(crate) threshold_db: f32,
    pub(crate) center_frequency: f32,
    pub(crate) curve_slope: f32,
    pub(crate) curve_curve: f32,
    pub(crate) points: [CurvePoint; MAX_THRESHOLD_CURVE_POINTS],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CurvePresetSource {
    BuiltIn,
    User,
}

impl CurvePresetSource {
    pub(crate) fn applies_anchor_parameters(self) -> bool {
        matches!(self, Self::User)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct CurvePresetEntry {
    pub(crate) preset: CurvePreset,
    pub(crate) source: CurvePresetSource,
    path: Option<PathBuf>,
}

#[derive(Debug, Default)]
pub(crate) struct LoadedCurvePresets {
    pub(crate) entries: Vec<CurvePresetEntry>,
    pub(crate) warnings: Vec<String>,
}

#[derive(Debug)]
pub(crate) enum CurvePresetError {
    BuiltInName(String),
    EmptyName,
    Io(io::Error),
    Serialize(serde_json::Error),
    Validation(String),
    NoUserPresetPath,
}

impl fmt::Display for CurvePresetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CurvePresetError::BuiltInName(name) => {
                write!(f, "Built-in preset '{name}' cannot be overwritten")
            }
            CurvePresetError::EmptyName => write!(f, "Preset name cannot be empty"),
            CurvePresetError::Io(err) => write!(f, "{err}"),
            CurvePresetError::Serialize(err) => write!(f, "{err}"),
            CurvePresetError::Validation(err) => write!(f, "{err}"),
            CurvePresetError::NoUserPresetPath => write!(f, "No user preset file is selected"),
        }
    }
}

impl From<io::Error> for CurvePresetError {
    fn from(err: io::Error) -> Self {
        CurvePresetError::Io(err)
    }
}

impl From<serde_json::Error> for CurvePresetError {
    fn from(err: serde_json::Error) -> Self {
        CurvePresetError::Serialize(err)
    }
}

#[derive(Debug, Clone, Copy)]
struct BuiltInCurvePreset {
    name: &'static str,
    threshold_db: f32,
    center_frequency: f32,
    curve_slope: f32,
    curve_curve: f32,
    points: [CurvePoint; MAX_THRESHOLD_CURVE_POINTS],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct UserCurvePresetFile {
    version: u32,
    name: String,
    threshold_db: f32,
    center_frequency: f32,
    curve_slope: f32,
    curve_curve: f32,
    points: Vec<UserCurvePoint>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
struct UserCurvePoint {
    enabled: bool,
    frequency: f32,
    offset_db: f32,
}

impl CurvePreset {
    pub(crate) fn built_in_entries() -> Vec<CurvePresetEntry> {
        BUILT_IN_CURVE_PRESETS
            .iter()
            .map(|preset| CurvePresetEntry {
                preset: (*preset).into(),
                source: CurvePresetSource::BuiltIn,
                path: None,
            })
            .collect()
    }

    pub(crate) fn is_built_in_name(name: &str) -> bool {
        BUILT_IN_CURVE_PRESETS
            .iter()
            .any(|preset| preset.name == name)
    }
}

impl From<BuiltInCurvePreset> for CurvePreset {
    fn from(preset: BuiltInCurvePreset) -> Self {
        Self {
            name: preset.name.to_owned(),
            threshold_db: preset.threshold_db,
            center_frequency: preset.center_frequency,
            curve_slope: preset.curve_slope,
            curve_curve: preset.curve_curve,
            points: preset.points,
        }
    }
}

impl From<CurvePoint> for UserCurvePoint {
    fn from(point: CurvePoint) -> Self {
        Self {
            enabled: point.enabled,
            frequency: point.frequency,
            offset_db: point.offset_db,
        }
    }
}

impl From<UserCurvePoint> for CurvePoint {
    fn from(point: UserCurvePoint) -> Self {
        Self {
            enabled: point.enabled,
            frequency: point.frequency,
            offset_db: point.offset_db,
        }
    }
}

impl From<&CurvePreset> for UserCurvePresetFile {
    fn from(preset: &CurvePreset) -> Self {
        Self {
            version: PRESET_FORMAT_VERSION,
            name: preset.name.clone(),
            threshold_db: preset.threshold_db,
            center_frequency: preset.center_frequency,
            curve_slope: preset.curve_slope,
            curve_curve: preset.curve_curve,
            points: preset.points.iter().copied().map(Into::into).collect(),
        }
    }
}

impl TryFrom<UserCurvePresetFile> for CurvePreset {
    type Error = CurvePresetError;

    fn try_from(file: UserCurvePresetFile) -> Result<Self, Self::Error> {
        if file.version != PRESET_FORMAT_VERSION {
            return Err(CurvePresetError::Validation(format!(
                "Unsupported preset format version {}",
                file.version
            )));
        }

        let name = normalize_name(&file.name)?;
        if file.points.len() != MAX_THRESHOLD_CURVE_POINTS {
            return Err(CurvePresetError::Validation(format!(
                "Expected {} curve points, found {}",
                MAX_THRESHOLD_CURVE_POINTS,
                file.points.len()
            )));
        }

        let mut points = [DISABLED_POINT; MAX_THRESHOLD_CURVE_POINTS];
        for (output, point) in points.iter_mut().zip(file.points) {
            *output = point.into();
        }

        let preset = CurvePreset {
            name,
            threshold_db: file.threshold_db,
            center_frequency: file.center_frequency,
            curve_slope: file.curve_slope,
            curve_curve: file.curve_curve,
            points,
        };
        validate_preset(&preset)?;
        Ok(preset)
    }
}

pub(crate) fn user_preset_dir() -> Option<PathBuf> {
    ProjectDirs::from("de", "Polarity-Music", "Polarity-SC-Dark")
        .map(|dirs| dirs.config_dir().join(USER_PRESET_FOLDER))
}

pub(crate) fn load_curve_presets() -> LoadedCurvePresets {
    let mut loaded = LoadedCurvePresets {
        entries: CurvePreset::built_in_entries(),
        warnings: Vec::new(),
    };

    if let Some(dir) = user_preset_dir() {
        let user_presets = load_user_presets_from_dir(&dir);
        loaded.entries.extend(user_presets.entries);
        loaded.warnings.extend(user_presets.warnings);
    }

    loaded
}

pub(crate) fn load_user_presets_from_dir(dir: &Path) -> LoadedCurvePresets {
    let mut loaded = LoadedCurvePresets::default();

    let read_dir = match fs::read_dir(dir) {
        Ok(read_dir) => read_dir,
        Err(err) if err.kind() == io::ErrorKind::NotFound => return loaded,
        Err(err) => {
            loaded
                .warnings
                .push(format!("Could not read {}: {err}", dir.display()));
            return loaded;
        }
    };

    for entry in read_dir {
        let entry = match entry {
            Ok(entry) => entry,
            Err(err) => {
                loaded
                    .warnings
                    .push(format!("Could not read preset directory entry: {err}"));
                continue;
            }
        };

        let path = entry.path();
        if path.extension().and_then(|extension| extension.to_str()) != Some("json") {
            continue;
        }

        match read_user_preset_file(&path) {
            Ok(preset) => loaded.entries.push(CurvePresetEntry {
                preset,
                source: CurvePresetSource::User,
                path: Some(path),
            }),
            Err(err) => loaded
                .warnings
                .push(format!("Could not load {}: {err}", path.display())),
        }
    }

    loaded.entries.sort_by(|left, right| {
        left.preset
            .name
            .cmp(&right.preset.name)
            .then_with(|| left.path.cmp(&right.path))
    });
    loaded.entries.dedup_by(|left, right| {
        left.source == CurvePresetSource::User
            && right.source == CurvePresetSource::User
            && left.preset.name == right.preset.name
    });

    loaded
}

pub(crate) fn save_user_preset_to_path(
    path: &Path,
    mut preset: CurvePreset,
) -> Result<CurvePresetEntry, CurvePresetError> {
    let Some(name) = path.file_stem().and_then(|name| name.to_str()) else {
        return Err(CurvePresetError::EmptyName);
    };

    preset.name = normalize_name(name)?;
    validate_preset(&preset)?;

    if CurvePreset::is_built_in_name(&preset.name) {
        return Err(CurvePresetError::BuiltInName(preset.name));
    }

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let path = with_json_extension(path);
    let json = serde_json::to_string_pretty(&UserCurvePresetFile::from(&preset))?;
    fs::write(&path, json)?;

    Ok(CurvePresetEntry {
        preset,
        source: CurvePresetSource::User,
        path: Some(path),
    })
}

pub(crate) fn delete_user_preset(entry: &CurvePresetEntry) -> Result<(), CurvePresetError> {
    if entry.source != CurvePresetSource::User {
        return Err(CurvePresetError::BuiltInName(entry.preset.name.clone()));
    }

    let Some(path) = entry.path.as_ref() else {
        return Err(CurvePresetError::NoUserPresetPath);
    };

    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(CurvePresetError::Io(err)),
    }
}

fn read_user_preset_file(path: &Path) -> Result<CurvePreset, CurvePresetError> {
    let json = fs::read_to_string(path)?;
    let file: UserCurvePresetFile = serde_json::from_str(&json)?;
    file.try_into()
}

fn validate_preset(preset: &CurvePreset) -> Result<(), CurvePresetError> {
    validate_range(
        "threshold",
        preset.threshold_db,
        THRESHOLD_MIN_DB,
        THRESHOLD_MAX_DB,
    )?;
    validate_range(
        "center frequency",
        preset.center_frequency,
        THRESHOLD_CURVE_MIN_FREQUENCY_HZ,
        THRESHOLD_CURVE_MAX_FREQUENCY_HZ,
    )?;
    validate_range(
        "curve slope",
        preset.curve_slope,
        CURVE_SLOPE_MIN,
        CURVE_SLOPE_MAX,
    )?;
    validate_range(
        "curve amount",
        preset.curve_curve,
        CURVE_CURVE_MIN,
        CURVE_CURVE_MAX,
    )?;

    for point in preset.points.iter().filter(|point| point.enabled) {
        validate_range(
            "point frequency",
            point.frequency,
            THRESHOLD_CURVE_MIN_FREQUENCY_HZ,
            THRESHOLD_CURVE_MAX_FREQUENCY_HZ,
        )?;
        validate_range(
            "point offset",
            point.offset_db,
            -THRESHOLD_CURVE_POINT_OFFSET_LIMIT_DB,
            THRESHOLD_CURVE_POINT_OFFSET_LIMIT_DB,
        )?;
    }

    Ok(())
}

fn validate_range(name: &str, value: f32, min: f32, max: f32) -> Result<(), CurvePresetError> {
    if value.is_finite() && (min..=max).contains(&value) {
        Ok(())
    } else {
        Err(CurvePresetError::Validation(format!(
            "{name} is out of range"
        )))
    }
}

fn normalize_name(name: &str) -> Result<String, CurvePresetError> {
    let name = name.trim();
    if name.is_empty() {
        Err(CurvePresetError::EmptyName)
    } else {
        Ok(name.to_owned())
    }
}

fn with_json_extension(path: &Path) -> PathBuf {
    if path.extension().is_some() {
        path.to_owned()
    } else {
        path.with_extension("json")
    }
}

const fn point(frequency: f32, offset_db: f32) -> CurvePoint {
    CurvePoint {
        enabled: true,
        frequency,
        offset_db,
    }
}

const DISABLED_POINT: CurvePoint = CurvePoint {
    enabled: false,
    frequency: 1_000.0,
    offset_db: 0.0,
};

const fn points<const N: usize>(
    enabled_points: [CurvePoint; N],
) -> [CurvePoint; MAX_THRESHOLD_CURVE_POINTS] {
    let mut output = [DISABLED_POINT; MAX_THRESHOLD_CURVE_POINTS];
    let mut index = 0;
    while index < N {
        output[index] = enabled_points[index];
        index += 1;
    }
    output
}

const BUILT_IN_CURVE_PRESETS: [BuiltInCurvePreset; 7] = [
    BuiltInCurvePreset {
        name: "Equal Loudness",
        threshold_db: -12.0,
        center_frequency: 1_000.0,
        curve_slope: 0.0,
        curve_curve: 0.0,
        points: points([
            point(35.0, 10.0),
            point(65.0, 6.0),
            point(120.0, 1.5),
            point(1_000.0, -1.0),
            point(3_500.0, -3.0),
            point(8_000.0, -1.5),
            point(15_000.0, 3.0),
        ]),
    },
    BuiltInCurvePreset {
        name: "Master Balanced",
        threshold_db: -12.0,
        center_frequency: 1_000.0,
        curve_slope: 0.0,
        curve_curve: 0.0,
        points: points([
            point(40.0, 3.0),
            point(90.0, 1.5),
            point(300.0, -0.5),
            point(1_000.0, 0.0),
            point(4_000.0, -1.0),
            point(10_000.0, 1.0),
        ]),
    },
    BuiltInCurvePreset {
        name: "Master Warm",
        threshold_db: -12.0,
        center_frequency: 900.0,
        curve_slope: -0.5,
        curve_curve: 0.0,
        points: points([
            point(45.0, 2.5),
            point(120.0, 0.5),
            point(350.0, -1.5),
            point(1_500.0, -0.5),
            point(5_000.0, 1.0),
            point(12_000.0, 3.5),
        ]),
    },
    BuiltInCurvePreset {
        name: "Bass Bus",
        threshold_db: -12.0,
        center_frequency: 120.0,
        curve_slope: 0.0,
        curve_curve: 0.0,
        points: points([
            point(35.0, 4.0),
            point(60.0, -2.5),
            point(110.0, -4.0),
            point(220.0, -2.0),
            point(700.0, 1.0),
            point(2_500.0, 3.0),
            point(8_000.0, 6.0),
        ]),
    },
    BuiltInCurvePreset {
        name: "Drum Bus",
        threshold_db: -12.0,
        center_frequency: 700.0,
        curve_slope: 0.0,
        curve_curve: 0.0,
        points: points([
            point(45.0, 3.0),
            point(80.0, -2.0),
            point(180.0, -1.0),
            point(500.0, 1.0),
            point(2_500.0, -2.5),
            point(6_000.0, -1.0),
            point(12_000.0, 2.0),
        ]),
    },
    BuiltInCurvePreset {
        name: "Pads",
        threshold_db: -12.0,
        center_frequency: 1_000.0,
        curve_slope: -0.25,
        curve_curve: 0.0,
        points: points([
            point(50.0, 5.0),
            point(120.0, 2.0),
            point(300.0, -2.0),
            point(1_200.0, -0.5),
            point(4_000.0, 0.5),
            point(10_000.0, 2.5),
        ]),
    },
    BuiltInCurvePreset {
        name: "Lead",
        threshold_db: -12.0,
        center_frequency: 1_500.0,
        curve_slope: 0.0,
        curve_curve: 0.0,
        points: points([
            point(60.0, 6.0),
            point(180.0, 2.5),
            point(700.0, -1.0),
            point(1_800.0, -2.0),
            point(4_000.0, -1.0),
            point(8_000.0, 2.5),
            point(14_000.0, 4.0),
        ]),
    },
];

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn built_in_presets_fit_existing_curve_parameter_limits() {
        for entry in CurvePreset::built_in_entries() {
            validate_preset(&entry.preset).expect("built-in preset should be valid");
        }
    }

    #[test]
    fn only_user_presets_apply_curve_anchor_parameters() {
        assert!(!CurvePresetSource::BuiltIn.applies_anchor_parameters());
        assert!(CurvePresetSource::User.applies_anchor_parameters());
    }

    #[test]
    fn serializes_and_deserializes_user_preset() {
        let preset = test_preset("Custom");
        let json = serde_json::to_string(&UserCurvePresetFile::from(&preset)).unwrap();
        let file: UserCurvePresetFile = serde_json::from_str(&json).unwrap();
        let restored = CurvePreset::try_from(file).unwrap();

        assert_eq!(restored.name, "Custom");
        assert_eq!(restored.points[0].frequency, 100.0);
        assert!(restored.points[0].enabled);
    }

    #[test]
    fn built_in_names_are_not_overwritten() {
        let dir = temp_dir("built-in-name");
        let err =
            save_user_preset_to_path(&dir.join("Lead.json"), test_preset("Ignored")).unwrap_err();

        assert!(matches!(err, CurvePresetError::BuiltInName(_)));
        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn user_presets_load_after_built_ins_and_are_sorted() {
        let dir = temp_dir("sorted-load");
        save_user_preset_to_path(&dir.join("Zulu.json"), test_preset("Ignored")).unwrap();
        save_user_preset_to_path(&dir.join("Alpha.json"), test_preset("Ignored")).unwrap();

        let mut loaded = CurvePreset::built_in_entries();
        loaded.extend(load_user_presets_from_dir(&dir).entries);

        let built_in_count = CurvePreset::built_in_entries().len();
        assert_eq!(
            loaded[built_in_count - 1].source,
            CurvePresetSource::BuiltIn
        );
        assert_eq!(loaded[built_in_count].preset.name, "Alpha");
        assert_eq!(loaded[built_in_count + 1].preset.name, "Zulu");

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn saving_same_dialog_path_overwrites_existing_file() {
        let dir = temp_dir("duplicate-overwrite");
        let path = dir.join("Custom.json");
        let first = save_user_preset_to_path(&path, test_preset("Ignored")).unwrap();
        let mut replacement = test_preset("Custom");
        replacement.threshold_db = -24.0;
        let second = save_user_preset_to_path(&path, replacement).unwrap();

        assert_eq!(first.path, second.path);

        let loaded = load_user_presets_from_dir(&dir);
        assert_eq!(loaded.entries.len(), 1);
        assert_eq!(loaded.entries[0].preset.threshold_db, -24.0);

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn save_dialog_path_uses_file_stem_as_preset_name() {
        let dir = temp_dir("dialog-path");
        let path = dir.join("Dialog Name.json");
        let entry = save_user_preset_to_path(&path, test_preset("Ignored")).unwrap();

        assert_eq!(entry.preset.name, "Dialog Name");
        assert_eq!(entry.path.as_deref(), Some(path.as_path()));
        assert!(path.exists());

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn delete_rejects_built_in_presets() {
        let entry = CurvePreset::built_in_entries()
            .into_iter()
            .next()
            .expect("missing built-in presets");
        let err = delete_user_preset(&entry).unwrap_err();

        assert!(matches!(err, CurvePresetError::BuiltInName(_)));
    }

    fn test_preset(name: &str) -> CurvePreset {
        CurvePreset {
            name: name.to_owned(),
            threshold_db: -12.0,
            center_frequency: 1_000.0,
            curve_slope: 0.0,
            curve_curve: 0.0,
            points: points([point(100.0, 1.0), point(1_000.0, -1.0)]),
        }
    }

    fn temp_dir(label: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("polarity-sc-dark-{label}-{nanos}"))
    }
}
