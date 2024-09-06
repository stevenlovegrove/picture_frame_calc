use egui::emath::Numeric;
use egui::{Color32, Context, Image, Mesh, Pos2, Rect, Shape, Stroke, Vec2};
use nalgebra::{DMatrix, DVector, Matrix3, Vector3};
use num_traits::Float;
use std::fmt::{self, Display};
use std::future::Future;
use std::str::FromStr;
use std::sync::mpsc::{channel, Receiver, Sender};

#[derive(Clone, Copy, Debug)]
pub struct Length<T: Float> {
    pub value: T,
    pub unit: LengthUnit,
}

impl<T: Float> Length<T> {
    pub fn new(value: T, unit: LengthUnit) -> Self {
        Self { value, unit }
    }

    pub fn to_unit(&self, unit: LengthUnit) -> Self {
        let mm = self.to_mm();
        Self::from_mm(mm, unit)
    }

    pub fn to_custom_unit(&self, unit_per_mm: T) -> T {
        let mm = self.to_mm();
        mm * unit_per_mm
    }

    pub fn to_mm(&self) -> T {
        match self.unit {
            LengthUnit::Meters => self.value * T::from(1000.0).unwrap(),
            LengthUnit::Centimeters => self.value * T::from(10.0).unwrap(),
            LengthUnit::Millimeters => self.value,
            LengthUnit::InchesDecimal | LengthUnit::InchesFractional => {
                self.value * T::from(25.4).unwrap()
            }
        }
    }

    pub fn from_mm(mm: T, unit: LengthUnit) -> Self {
        let value = match unit {
            LengthUnit::Meters => mm / T::from(1000.0).unwrap(),
            LengthUnit::Centimeters => mm / T::from(10.0).unwrap(),
            LengthUnit::Millimeters => mm,
            LengthUnit::InchesDecimal | LengthUnit::InchesFractional => mm / T::from(25.4).unwrap(),
        };
        Self { value, unit }
    }

    pub fn cast<U: Float>(self) -> Length<U> {
        Length::<U> {
            value: U::from(self.to_mm()).unwrap(),
            unit: self.unit,
        }
    }
}

impl<T: Float> std::ops::Add for Length<T> {
    type Output = Length<T>;

    fn add(self, other: Self) -> Self::Output {
        let other_mm = other.to_mm();
        let result_mm = self.to_mm() + other_mm;
        Length::from_mm(result_mm, self.unit)
    }
}

impl<T: Float> std::ops::Sub for Length<T> {
    type Output = Length<T>;

    fn sub(self, other: Self) -> Self::Output {
        let other_mm = other.to_mm();
        let result_mm = self.to_mm() - other_mm;
        Length::from_mm(result_mm, self.unit)
    }
}

impl<T: Float> std::ops::Mul<T> for Length<T> {
    type Output = Length<T>;

    fn mul(self, scalar: T) -> Self::Output {
        Self {
            value: self.value * scalar,
            unit: self.unit,
        }
    }
}

impl<T: Float> std::ops::Div<T> for Length<T> {
    type Output = Length<T>;

    fn div(self, scalar: T) -> Self::Output {
        Self {
            value: self.value / scalar,
            unit: self.unit,
        }
    }
}

#[derive(Clone, Copy)]
pub struct LengthVec<T: Float, const N: usize> {
    values: [Length<T>; N],
}

impl<T: Float, const N: usize> LengthVec<T, N> {
    pub fn new(values: [T; N], unit: LengthUnit) -> Self {
        let values = values.map(|v| Length::new(v, unit));
        Self { values }
    }

    pub fn to_unit(&self, unit: LengthUnit) -> Self {
        Self {
            values: self.values.map(|v| v.to_unit(unit)),
        }
    }

    pub fn to_mm(&self) -> [T; N] {
        self.values.map(|v| v.to_mm())
    }

    pub fn from_mm(mm: [T; N], unit: LengthUnit) -> Self {
        Self {
            values: mm.map(|v| Length::from_mm(v, unit)),
        }
    }

    pub fn to_custom_unit(&self, unit_per_mm: T) -> [T; N] {
        self.values.map(|v| v.to_custom_unit(unit_per_mm))
    }

    pub fn cast<U: Float>(self) -> LengthVec<U, N> {
        LengthVec::<U, N> {
            values: self.values.map(|v| v.cast()),
        }
    }
}

impl<T: Float, const N: usize> std::ops::Add for LengthVec<T, N> {
    type Output = LengthVec<T, N>;

    fn add(self, other: Self) -> Self::Output {
        Self {
            values: std::array::from_fn(|i| self.values[i] + other.values[i]),
        }
    }
}

impl<T: Float, const N: usize> std::ops::Sub for LengthVec<T, N> {
    type Output = LengthVec<T, N>;

    fn sub(self, other: Self) -> Self::Output {
        Self {
            values: std::array::from_fn(|i| self.values[i] - other.values[i]),
        }
    }
}

impl<T: Float, const N: usize> std::ops::Mul<T> for LengthVec<T, N> {
    type Output = LengthVec<T, N>;

    fn mul(self, scalar: T) -> Self::Output {
        Self {
            values: self.values.map(|v| v * scalar),
        }
    }
}

impl<T: Float, const N: usize> std::ops::Div<T> for LengthVec<T, N> {
    type Output = LengthVec<T, N>;

    fn div(self, scalar: T) -> Self::Output {
        Self {
            values: self.values.map(|v| v / scalar),
        }
    }
}

#[derive(Clone)]
pub struct PictureFrame {
    texture: Option<egui::TextureHandle>,
    matte_color: Color32,
    frame_color: Color32,
    picture_size: LengthVec<f64, 2>,
    glass_size: LengthVec<f64, 2>,
    glass_frame_lip: Length<f64>,
    glass_frame_tol: Length<f64>,
    picture_offset: LengthVec<f64, 2>,
    matte_hole_size: LengthVec<f64, 2>,
    frame_width: Length<f64>,
    frame_depth: Length<f64>,
    thickness_glass: Length<f64>,
    thickness_matte: Length<f64>,
    thickness_backing: Length<f64>,
}

pub struct PictureFrameApp {
    img_channel: (Sender<Vec<u8>>, Receiver<Vec<u8>>),
    quad_points: Option<[Pos2; 4]>,
    pix_per_mm: f32,
    frame: Option<PictureFrame>,
}

impl Default for PictureFrameApp {
    fn default() -> Self {
        Self {
            img_channel: channel(),
            quad_points: None,
            pix_per_mm: 2.0,
            frame: None,
        }
    }
}

impl PictureFrameApp {
    /// Called once before the first frame.
    pub fn new(_: &eframe::CreationContext<'_>) -> Self {
        let app = PictureFrameApp::default();
        let sender = app.img_channel.0.clone();

        // Load the sample image
        #[cfg(not(target_arch = "wasm32"))]
        {
            let sample_path = std::path::Path::new("assets/sample.jpg");
            if let Ok(image_data) = std::fs::read(sample_path) {
                // app.load_image(image_data, cc.egui_ctx.clone());
                let _ = sender.send(image_data);
            }
        }

        #[cfg(target_arch = "wasm32")]
        {
            wasm_bindgen_futures::spawn_local(async move {
                let request = ehttp::Request::get("assets/sample.jpg");
                if let Ok(response) = ehttp::fetch_async(request).await {
                    let _ = sender.send(response.bytes);
                }
            });
        }

        app
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LengthUnit {
    Meters,
    Centimeters,
    Millimeters,
    InchesFractional,
    InchesDecimal,
}

impl fmt::Display for LengthUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LengthUnit::Meters => write!(f, "m"),
            LengthUnit::Centimeters => write!(f, "cm"),
            LengthUnit::Millimeters => write!(f, "mm"),
            LengthUnit::InchesDecimal => write!(f, ".in"),
            LengthUnit::InchesFractional => write!(f, "in"),
        }
    }
}

impl LengthUnit {
    fn to_mm_factor(&self) -> f64 {
        match self {
            LengthUnit::Meters => 1000.0,
            LengthUnit::Centimeters => 10.0,
            LengthUnit::Millimeters => 1.0,
            LengthUnit::InchesDecimal | LengthUnit::InchesFractional => 25.4,
        }
    }
}

pub struct ImperialDragValue<'a, T: Float> {
    value: &'a mut Length<T>,
    speed: T,
    _id: String,
}

impl<'a, T: Float + Display + FromStr + Numeric> ImperialDragValue<'a, T> {
    pub fn new(value: &'a mut Length<T>, id: impl Into<String>) -> Self {
        Self {
            value,
            speed: T::from(0.1).unwrap(),
            _id: id.into(),
        }
    }

    pub fn speed(mut self, speed_mm: T) -> Self {
        self.speed = speed_mm / T::from(self.value.unit.to_mm_factor()).unwrap();
        self
    }

    fn format_value(&self) -> String {
        match self.value.unit {
            LengthUnit::Meters => format!("{:.3}", self.value.value),
            LengthUnit::Centimeters => format!("{:.2}", self.value.value),
            LengthUnit::Millimeters => format!("{:.1}", self.value.value),
            LengthUnit::InchesDecimal => format!("{:.3}", self.value.value),
            LengthUnit::InchesFractional => {
                let inches = self.value.value;
                let whole_inches = inches.floor();
                let fraction = inches - whole_inches;
                let denominator = 32;
                let numerator = (fraction * T::from(denominator).unwrap())
                    .round()
                    .to_i32()
                    .unwrap();

                if numerator == 0 {
                    format!("{}", whole_inches)
                } else if numerator == denominator {
                    format!("{}", whole_inches + T::from(1.0).unwrap())
                } else {
                    // Simplify the fraction
                    let gcd = gcd(numerator, denominator);
                    let simplified_numerator = numerator / gcd;
                    let simplified_denominator = denominator / gcd;
                    format!(
                        "{} {}/{}",
                        whole_inches, simplified_numerator, simplified_denominator
                    )
                }
            }
        }
    }

    fn parse_value(s: &str) -> Option<T> {
        let s = s.trim();
        let parts: Vec<&str> = s.split_whitespace().collect();

        if parts.is_empty() {
            return None;
        }

        let whole = if parts[0].contains('.') {
            parts[0].parse::<T>().ok()?
        } else {
            T::from(parts[0].parse::<i32>().ok()?).unwrap()
        };

        if parts.len() == 1 {
            return Some(whole);
        }

        let fraction_parts: Vec<&str> = parts[1].split('/').collect();
        if fraction_parts.len() != 2 {
            return None;
        }

        let numerator = fraction_parts[0].parse::<T>().ok()?;
        let denominator = fraction_parts[1].parse::<T>().ok()?;

        Some(whole + numerator / denominator)
    }

    pub fn ui(self, ui: &mut egui::Ui) -> egui::Response {
        let mut changed = false;

        ui.scope(|ui| {
            // Reduce the item spacing for this scope
            ui.spacing_mut().item_spacing.x = 1.0;

            ui.horizontal(|ui| {
                let mut display_value = self.value.value;

                let drag_response = ui.add(
                    egui::DragValue::new(&mut display_value)
                        .speed(self.speed.to_f64())
                        .custom_formatter(|_, _| self.format_value())
                        .custom_parser(|s| Self::parse_value(s).map(|v| v.to_f64())),
                );

                if drag_response.changed() {
                    changed = true;
                    self.value.value = display_value;
                }

                let prev = self.value.clone();

                ui.menu_button(self.value.unit.to_string(), |ui| {
                    for unit in [
                        LengthUnit::Meters,
                        LengthUnit::Centimeters,
                        LengthUnit::Millimeters,
                        LengthUnit::InchesDecimal,
                        LengthUnit::InchesFractional,
                    ] {
                        if ui
                            .selectable_label(self.value.unit == unit, unit.to_string())
                            .clicked()
                        {
                            if self.value.unit != unit {
                                changed = true;
                                let mm = prev.to_mm();
                                *self.value = Length::from_mm(mm, unit);
                            }
                            ui.close_menu();
                        }
                    }
                });

                drag_response
            })
            .inner
        })
        .inner
    }
}

// Add this helper function outside of the impl block
fn gcd(mut a: i32, mut b: i32) -> i32 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a.abs()
}

#[derive(Clone, Copy, Debug)]
pub enum FrameSide {
    Horizontal,
    Vertical,
}

impl eframe::App for PictureFrameApp {
    fn update(&mut self, ctx: &Context, _frame: &mut eframe::Frame) {
        // Process image data when it comes in
        if let Ok(image_data) = self.img_channel.1.try_recv() {
            if let Ok(image) = image::load_from_memory(&image_data) {
                let size = [image.width() as _, image.height() as _];
                let image_buffer = image.to_rgba8();
                let pixels = image_buffer.as_flat_samples();
                let color_image = egui::ColorImage::from_rgba_unmultiplied(size, pixels.as_slice());
                // default quad points to be the size of the image
                self.quad_points = Some([
                    Pos2::new(0.0, 0.0),
                    Pos2::new(image.width() as f32, 0.0),
                    Pos2::new(image.width() as f32, image.height() as f32),
                    Pos2::new(0.0, image.height() as f32),
                ]);
                let aspect_ratio = image.width() as f64 / image.height() as f64;
                self.frame = Some(PictureFrame {
                    texture: Some(ctx.load_texture(
                        "uploaded_image",
                        color_image,
                        egui::TextureOptions::default(),
                    )),
                    matte_color: Color32::WHITE,
                    frame_color: Color32::from_rgb(139, 69, 19),
                    picture_size: LengthVec::new(
                        [100.0, 100.0 / aspect_ratio],
                        LengthUnit::Millimeters,
                    ),
                    glass_size: LengthVec::new(
                        [120.0, 120.0 / aspect_ratio],
                        LengthUnit::Millimeters,
                    ),
                    glass_frame_lip: Length::new(5.0, LengthUnit::Millimeters),
                    glass_frame_tol: Length::new(0.0, LengthUnit::Millimeters),
                    picture_offset: LengthVec::new([0.0, 0.0], LengthUnit::Millimeters),
                    matte_hole_size: LengthVec::new(
                        [100.0, 100.0 / aspect_ratio],
                        LengthUnit::Millimeters,
                    ),
                    frame_width: Length::new(20.0, LengthUnit::Millimeters),
                    frame_depth: Length::new(20.0, LengthUnit::Millimeters),
                    thickness_glass: Length::new(3.0 / 32.0, LengthUnit::InchesFractional),
                    thickness_matte: Length::new(3.0 / 32.0, LengthUnit::InchesFractional),
                    thickness_backing: Length::new(3.0 / 32.0, LengthUnit::InchesFractional),
                });
            }
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                if ui.button("📂 Open image file").clicked() {
                    let sender = self.img_channel.0.clone();
                    let task = rfd::AsyncFileDialog::new()
                        .add_filter("Image", &["png", "jpg", "jpeg", "gif", "bmp"])
                        .pick_file();
                    let ctx = ui.ctx().clone();
                    execute(async move {
                        let file = task.await;
                        if let Some(file) = file {
                            let image_data = file.read().await;
                            let _ = sender.send(image_data);
                            ctx.request_repaint();
                        }
                    });
                }

                if let Some(frame) = &mut self.frame {
                    let mut frame = frame.clone();
                    let mut quad_points = self.quad_points.clone();

                    ui.heading("Configuration");

                    egui::Grid::new("frame_dimensions")
                        .num_columns(5)
                        .spacing([2.0, 4.0])
                        .min_col_width(0.0)
                        .show(ui, |ui| {
                            ui.label("Picture size");
                            ImperialDragValue::new(&mut frame.picture_size.values[0], "width")
                                .speed(0.1)
                                .ui(ui);
                            ui.label("x");
                            ImperialDragValue::new(&mut frame.picture_size.values[1], "height")
                                .speed(0.1)
                                .ui(ui);
                            standard_size_menu_button(ui, &mut frame.picture_size);
                            ui.end_row();

                            ui.label("Glass size");
                            ImperialDragValue::new(&mut frame.glass_size.values[0], "width")
                                .speed(0.1)
                                .ui(ui);
                            ui.label("x");
                            ImperialDragValue::new(&mut frame.glass_size.values[1], "height")
                                .speed(0.1)
                                .ui(ui);
                            standard_size_menu_button(ui, &mut frame.glass_size);
                            ui.end_row();

                            ui.label("Matte hole size:");
                            ImperialDragValue::new(
                                &mut frame.matte_hole_size.values[0],
                                "matte_hole_size_width",
                            )
                            .speed(0.1)
                            .ui(ui);
                            ui.label("x");
                            ImperialDragValue::new(
                                &mut frame.matte_hole_size.values[1],
                                "matte_hole_size_height",
                            )
                            .speed(0.1)
                            .ui(ui);
                            standard_size_menu_button(ui, &mut frame.matte_hole_size);
                            ui.end_row();

                            ui.label("Picture offset");
                            ImperialDragValue::new(
                                &mut frame.picture_offset.values[0],
                                "picture_offset_x",
                            )
                            .speed(0.1)
                            .ui(ui);
                            ui.label("x");
                            ImperialDragValue::new(
                                &mut frame.picture_offset.values[1],
                                "picture_offset_y",
                            )
                            .speed(0.1)
                            .ui(ui);
                            if ui.button("🔄").clicked() {
                                frame.picture_offset.values[0] =
                                    Length::new(0.0, frame.picture_offset.values[0].unit);
                                frame.picture_offset.values[1] =
                                    Length::new(0.0, frame.picture_offset.values[1].unit);
                            }
                            ui.end_row();

                            ui.label("Frame (width x depth):");
                            ImperialDragValue::new(&mut frame.frame_width, "frame_width")
                                .speed(0.1)
                                .ui(ui);
                            ui.label("x");
                            ImperialDragValue::new(&mut frame.frame_depth, "frame_depth")
                                .speed(0.1)
                                .ui(ui);
                            ui.end_row();

                            ui.label("Glass frame lip:");
                            ImperialDragValue::new(&mut frame.glass_frame_lip, "glass_frame_lip")
                                .speed(0.1)
                                .ui(ui);
                            ui.end_row();

                            ui.label("Glass frame tolerance:");
                            ImperialDragValue::new(&mut frame.glass_frame_tol, "glass_frame_tol")
                                .speed(0.1)
                                .ui(ui);
                            ui.end_row();

                            ui.label("Frame color:");
                            ui.color_edit_button_srgba(&mut frame.frame_color);
                            ui.end_row();

                            ui.label("Matte color:");
                            ui.color_edit_button_srgba(&mut frame.matte_color);
                            ui.end_row();

                            ui.label("Thickness glass:");
                            ImperialDragValue::new(&mut frame.thickness_glass, "thickness_glass")
                                .speed(0.1)
                                .ui(ui);
                            ui.end_row();

                            ui.label("Thickness matte:");
                            ImperialDragValue::new(&mut frame.thickness_matte, "thickness_matte")
                                .speed(0.1)
                                .ui(ui);
                            ui.end_row();

                            ui.label("Thickness backing:");
                            ImperialDragValue::new(
                                &mut frame.thickness_backing,
                                "thickness_backing",
                            )
                            .speed(0.1)
                            .ui(ui);
                            ui.end_row();
                        });

                    ui.heading("Preview");

                    ui.horizontal(|ui| {
                        if let Some(texture) = &frame.texture {
                            // Calculate the scaled image size
                            let max_size = 400.0; // Reduced maximum width or height
                            let image_size =
                                Vec2::new(texture.size()[0] as f32, texture.size()[1] as f32);
                            let scale = (max_size / image_size.x)
                                .min(max_size / image_size.y)
                                .min(1.0);
                            let scaled_size = image_size * scale;

                            // Display the original image
                            let (rect, _) =
                                ui.allocate_exact_size(scaled_size, egui::Sense::drag());
                            ui.put(rect, Image::new(texture).fit_to_exact_size(scaled_size));

                            if let Some(updated_quad_points) = quad_points.as_mut() {
                                // Draw the quadrilateral
                                let quad_points =
                                    updated_quad_points.map(|p| rect.min + p.to_vec2() * scale);
                                ui.painter().add(Shape::closed_line(
                                    quad_points.to_vec(),
                                    Stroke::new(2.0, egui::Color32::RED),
                                ));

                                // Handle dragging of quadrilateral points
                                for (i, point) in updated_quad_points.iter_mut().enumerate() {
                                    let point_pos = rect.min + point.to_vec2() * scale;
                                    let point_rect =
                                        Rect::from_center_size(point_pos, Vec2::splat(10.0));
                                    let id = ui.id().with(i);
                                    let response = ui.interact(point_rect, id, egui::Sense::drag());

                                    if response.dragged() {
                                        *point += response.drag_delta() / scale;
                                        *point = point.clamp(Pos2::ZERO, image_size.to_pos2());
                                    }

                                    ui.painter().circle(
                                        point_pos,
                                        5.0,
                                        egui::Color32::RED,
                                        Stroke::new(1.0, egui::Color32::WHITE),
                                    );
                                }
                            }
                        }

                        if let Some(frame) = &self.frame {
                            if let Some(updated_quad_points) = quad_points {
                                let src_points =
                                    updated_quad_points.map(|p| (p.x as f64, p.y as f64));
                                let dst_points = [
                                    (0.0, 0.0),
                                    (frame.picture_size.values[0].to_mm() as f64, 0.0),
                                    (
                                        frame.picture_size.values[0].to_mm() as f64,
                                        frame.picture_size.values[1].to_mm() as f64,
                                    ),
                                    (0.0, frame.picture_size.values[1].to_mm() as f64),
                                ];
                                let homography_result =
                                    compute_homography_from_points(&src_points, &dst_points);

                                match &homography_result {
                                    Ok(mm_from_pix) => {
                                        render_picture_frame(
                                            ui,
                                            &frame,
                                            self.pix_per_mm,
                                            mm_from_pix,
                                        );
                                    }
                                    Err(e) => {
                                        ui.label(format!("Error computing homography: {}", e));
                                    }
                                }
                            }
                        }
                    });

                    if let Some(frame) = &self.frame {
                        // Render bottom side
                        ui.heading("Cut List");
                        ui.label("Horizontal side");
                        render_frame_side_quad(ui, frame, self.pix_per_mm, FrameSide::Horizontal);
                        ui.label("Vertical side");
                        render_frame_side_quad(ui, frame, self.pix_per_mm, FrameSide::Vertical);
                        ui.label("Profile");
                        render_frame_profile(ui, frame, self.pix_per_mm);
                        ui.label("Matte");
                        render_matte_profile(ui, frame, self.pix_per_mm);
                    }

                    // Update the quad points in self after the ui.horizontal() block
                    self.quad_points = quad_points;
                    self.frame = Some(frame);
                } else {
                    ui.label("Upload an image to get started.");
                }
            });
        });
    }
}

fn render_picture_frame(
    ui: &mut egui::Ui,
    frame: &PictureFrame,
    pix_per_mm: f32,
    mm_from_pix: &Matrix3<f64>,
) -> egui::Response {
    let total_size_mm = (
        frame.glass_size.values[0].to_mm()
            + frame.glass_frame_tol.to_mm() * 2.0
            + 2.0 * (frame.frame_width.to_mm() - frame.glass_frame_lip.to_mm()),
        frame.glass_size.values[1].to_mm()
            + frame.glass_frame_tol.to_mm() * 2.0
            + 2.0 * (frame.frame_width.to_mm() - frame.glass_frame_lip.to_mm()),
    );
    let total_size_px = egui::Vec2::new(
        (total_size_mm.0 * pix_per_mm as f64) as f32,
        (total_size_mm.1 * pix_per_mm as f64) as f32,
    );
    // let frame_width_px = frame.frame_width.to_mm() * pix_per_mm as f64;

    // Allocate space for the entire framed image
    let (total_rect, response) = ui.allocate_exact_size(total_size_px, egui::Sense::hover());

    // Outer frame
    ui.painter().rect_filled(total_rect, 0.0, frame.frame_color);

    let picture_offset_px = egui::Vec2::new(
        (frame.picture_offset.values[0].to_mm() * pix_per_mm as f64) as f32,
        (frame.picture_offset.values[1].to_mm() * pix_per_mm as f64) as f32,
    );
    let picture_size_pix = egui::Vec2::new(
        (frame.picture_size.values[0].to_mm() * pix_per_mm as f64) as f32,
        (frame.picture_size.values[1].to_mm() * pix_per_mm as f64) as f32,
    );

    let center_offset = (total_rect.size() - picture_size_pix) / 2.0;

    // Calculate the inner rect for the warped image
    let warped_rect = Rect {
        min: total_rect.min + center_offset + picture_offset_px,
        max: total_rect.min + center_offset + picture_offset_px + picture_size_pix,
    };

    // Render the warped image
    render_warped_image(ui, warped_rect, &frame, mm_from_pix);

    // Render the matte (over the image)
    let matte_outer_rect = Rect {
        min: total_rect.min
            + egui::Vec2::new(
                frame.frame_width.to_mm() as f32 * pix_per_mm,
                frame.frame_width.to_mm() as f32 * pix_per_mm,
            ),
        max: total_rect.max
            - egui::Vec2::new(
                frame.frame_width.to_mm() as f32 * pix_per_mm,
                frame.frame_width.to_mm() as f32 * pix_per_mm,
            ),
    };
    let matte_hole_size = egui::Vec2::new(
        frame.matte_hole_size.values[0].to_mm() as f32 * pix_per_mm,
        frame.matte_hole_size.values[1].to_mm() as f32 * pix_per_mm,
    );
    render_rect_with_hole(ui, matte_outer_rect, matte_hole_size, frame.matte_color);

    response
}

fn render_rect_with_hole(ui: &mut egui::Ui, outer_rect: Rect, hole_size: Vec2, color: Color32) {
    let hole_rect = Rect::from_center_size(outer_rect.center(), hole_size);

    // Top rectangle
    ui.painter().rect_filled(
        Rect::from_min_max(outer_rect.min, hole_rect.right_top()),
        0.0,
        color,
    );

    // Bottom rectangle
    ui.painter().rect_filled(
        Rect::from_min_max(
            Pos2::new(outer_rect.left(), hole_rect.bottom()),
            outer_rect.max,
        ),
        0.0,
        color,
    );

    // Left rectangle
    ui.painter().rect_filled(
        Rect::from_min_max(outer_rect.left_top(), hole_rect.left_bottom()),
        0.0,
        color,
    );

    // Right rectangle
    ui.painter().rect_filled(
        Rect::from_min_max(
            Pos2::new(hole_rect.right(), outer_rect.top()),
            outer_rect.right_bottom(),
        ),
        0.0,
        color,
    );
}

fn render_frame_side_quad(
    ui: &mut egui::Ui,
    frame: &PictureFrame,
    pix_per_mm: f32,
    side: FrameSide,
) -> egui::Response {
    // Calculate dimensions
    let frame_width_px = frame.frame_width.to_mm() as f32 * pix_per_mm;
    let glass_edge_to_frame_edge_mm = frame.frame_width.to_mm() - frame.glass_frame_lip.to_mm();
    let frame_length_mm = match side {
        FrameSide::Horizontal => {
            frame.glass_size.values[0].to_mm()
                + frame.glass_frame_tol.to_mm() * 2.0
                + 2.0 * glass_edge_to_frame_edge_mm
        }
        FrameSide::Vertical => {
            frame.glass_size.values[1].to_mm()
                + frame.glass_frame_tol.to_mm() * 2.0
                + 2.0 * glass_edge_to_frame_edge_mm
        }
    };

    let frame_length_px = frame_length_mm as f32 * pix_per_mm;

    // Calculate the size of the area we need
    let padding = 20.0;
    let padding_vec = Vec2::new(padding, padding);
    let widget_size = Vec2::new(
        frame_length_px + 2.0 * padding,
        frame_width_px + 2.0 * padding,
    );
    let (rect, response) = ui.allocate_exact_size(widget_size, egui::Sense::hover());
    let origin = rect.left_top() + padding_vec;

    // Define the four corners of the trapezoid
    let points = [
        origin + Vec2::new(0.0, 0.0),
        origin + Vec2::new(frame_length_px, 0.0),
        origin + Vec2::new(frame_length_px - frame_width_px, frame_width_px),
        origin + Vec2::new(frame_width_px, frame_width_px),
        origin + Vec2::new(0.0, 0.0),
    ];

    // Render the outline
    ui.painter().add(Shape::line(
        points.to_vec(),
        Stroke::new(1.0, Color32::BLACK),
    ));

    // Render the gap recess line
    let frame_lip_px = frame.glass_frame_lip.to_mm() as f32 * pix_per_mm;
    if frame_lip_px > 0.0 {
        ui.painter().add(Shape::dotted_line(
            &[
                points[3] + Vec2::new(-frame_lip_px, -frame_lip_px),
                points[2] + Vec2::new(frame_lip_px, -frame_lip_px),
            ],
            Color32::BLACK,
            2.0,
            0.5,
        ));
    }

    // Render dimension lines and labels
    let text_style = egui::TextStyle::Small;
    let font_id = ui.style().text_styles[&text_style].clone();

    // Long side
    render_dimension_line(
        ui,
        points[0],
        points[1],
        padding / 2.0,
        &format!("{:.1} {}", frame_length_mm, "mm"),
        &font_id,
    );

    // short side
    render_dimension_line(
        ui,
        points[3],
        points[2],
        -padding / 2.0,
        &format!(
            "{:.1} {}",
            frame_length_mm - 2.0 * frame.frame_width.to_mm(),
            "mm"
        ),
        &font_id,
    );

    // Frame width
    render_dimension_line(
        ui,
        points[1],
        points[1] + Vec2::new(0.0, frame_width_px),
        padding / 2.0,
        &format!("{:.1} {}", frame.frame_width.to_mm(), "mm"),
        &font_id,
    );

    response
}

fn render_dimension_line(
    ui: &mut egui::Ui,
    start: Pos2,
    end: Pos2,
    offset: f32,
    label: &str,
    font_id: &egui::FontId,
) {
    let direction = (end - start).normalized();
    let normal = direction.rot90();
    let line_start = start + normal * offset;
    let line_end = end + normal * offset;

    // Draw perpendicular lines
    ui.painter()
        .line_segment([start, line_start], Stroke::new(1.0, Color32::LIGHT_GRAY));
    ui.painter()
        .line_segment([end, line_end], Stroke::new(1.0, Color32::LIGHT_GRAY));

    let line_color = Color32::from_gray(150);

    // Draw the main line
    ui.painter()
        .line_segment([line_start, line_end], Stroke::new(1.0, line_color));

    // Draw arrow caps
    let arrow_size = 3.0;
    let arrow1 = (direction + normal).normalized() * arrow_size;
    let arrow2 = (direction - normal).normalized() * arrow_size;
    ui.painter().line_segment(
        [line_start, line_start + arrow1],
        Stroke::new(1.0, line_color),
    );
    ui.painter().line_segment(
        [line_start, line_start + arrow2],
        Stroke::new(1.0, line_color),
    );
    ui.painter()
        .line_segment([line_end, line_end - arrow1], Stroke::new(1.0, line_color));
    ui.painter()
        .line_segment([line_end, line_end - arrow2], Stroke::new(1.0, line_color));

    // Draw label
    let galley = ui
        .painter()
        .layout_no_wrap(label.to_string(), font_id.clone(), line_color);
    let label_pos = (line_start + line_end.to_vec2()) / 2.0
        + offset.signum()
            * (rect_width_in_direction(galley.rect, normal) / 2.0 + offset.abs() / 2.0)
            * normal
        - galley.rect.size() / 2.0;
    ui.painter().galley(label_pos, galley, line_color);
}

fn rect_width_in_direction(rect: Rect, normalized_direction: Vec2) -> f32 {
    normalized_direction.dot(rect.size()).abs()
}

fn render_warped_image(
    ui: &mut egui::Ui,
    rect: Rect,
    frame: &PictureFrame,
    mm_from_pix: &Matrix3<f64>,
) {
    if let Some(texture) = &frame.texture {
        let mut mesh = Mesh::default();

        let steps = 20; // Increase for higher quality, decrease for better performance
        let step_size = Vec2::new(rect.width() / steps as f32, rect.height() / steps as f32);

        // 1. Transform from egui window pos to picture coords in mm
        let mm_from_window = Matrix3::new(
            frame.picture_size.values[0].to_mm() as f64 / rect.width() as f64,
            0.0,
            -frame.picture_size.values[0].to_mm() as f64 * rect.min.x as f64 / rect.width() as f64,
            0.0,
            frame.picture_size.values[1].to_mm() as f64 / rect.height() as f64,
            -frame.picture_size.values[1].to_mm() as f64 * rect.min.y as f64 / rect.height() as f64,
            0.0,
            0.0,
            1.0,
        );

        // 2. Transform from mm to pixel coords in source image
        // This is our input homography (mm_from_pix)

        // 3. Transform from pixel coords to UV texture coords
        let uv_from_pix = Matrix3::new(
            1.0 / texture.size()[0] as f64,
            0.0,
            0.0,
            0.0,
            1.0 / texture.size()[1] as f64,
            0.0,
            0.0,
            0.0,
            1.0,
        );

        // Combine all transformations
        let uv_from_window =
            uv_from_pix * mm_from_pix.try_inverse().unwrap_or(Matrix3::identity()) * mm_from_window;

        for y in 0..=steps {
            for x in 0..=steps {
                let screen_pos =
                    rect.min + Vec2::new(x as f32 * step_size.x, y as f32 * step_size.y);

                // Apply combined transform to get texture coordinates
                let p =
                    uv_from_window * Vector3::new(screen_pos.x as f64, screen_pos.y as f64, 1.0);
                let uv = Pos2::new((p.x / p.z) as f32, (p.y / p.z) as f32);

                mesh.vertices.push(egui::epaint::Vertex {
                    pos: screen_pos,
                    uv,
                    color: Color32::WHITE,
                });

                if x < steps && y < steps {
                    let idx = (y * (steps + 1) + x) as u32;
                    mesh.indices.push(idx);
                    mesh.indices.push(idx + 1);
                    mesh.indices.push(idx + steps + 1);
                    mesh.indices.push(idx + 1);
                    mesh.indices.push(idx + steps + 2);
                    mesh.indices.push(idx + steps + 1);
                }
            }
        }

        ui.painter().add(Shape::Mesh(egui::epaint::Mesh {
            texture_id: texture.id(),
            ..mesh
        }));
    }
}

fn compute_homography_from_points(
    src_points: &[(f64, f64); 4],
    dst_points: &[(f64, f64); 4],
) -> anyhow::Result<Matrix3<f64>> {
    // Construct the 8x8 matrix A and 8x1 vector b
    let mut a = DMatrix::<f64>::zeros(8, 8);
    let mut b = DVector::<f64>::zeros(8);

    for i in 0..4 {
        let (x, y) = src_points[i];
        let (u, v) = dst_points[i];
        let row1 = [x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y];
        let row2 = [0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y];
        a.set_row(2 * i, &row1.into());
        a.set_row(2 * i + 1, &row2.into());
        b[2 * i] = u;
        b[2 * i + 1] = v;
    }

    // Solve the system
    match a.svd(true, true).solve(&b, 1e-15) {
        Ok(x) => {
            // Construct the homography matrix
            Ok(Matrix3::new(
                x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], 1.0,
            ))
        }
        Err(str) => Err(anyhow::anyhow!("Failed to solve the system: {}", str)),
    }
}

fn render_frame_profile(
    ui: &mut egui::Ui,
    frame: &PictureFrame,
    pix_per_mm: f32,
) -> egui::Response {
    // Calculate dimensions
    let frame_width_px = frame.frame_width.to_mm() as f32 * pix_per_mm;
    let frame_depth_px = frame.frame_depth.to_mm() as f32 * pix_per_mm;
    let glass_frame_lip_px =
        (frame.glass_frame_lip.to_mm() + frame.glass_frame_tol.to_mm()) as f32 * pix_per_mm;
    let gap_depth_mm = frame.frame_depth.to_mm()
        - frame.thickness_glass.to_mm()
        - frame.thickness_matte.to_mm()
        - frame.thickness_backing.to_mm();
    let gap_depth_px = gap_depth_mm as f32 * pix_per_mm;

    // Calculate the size of the area we need
    let padding = 20.0;
    let padding_vec = Vec2::new(padding, padding);
    let widget_size = Vec2::new(
        frame_width_px + 2.0 * padding,
        frame_depth_px + 2.0 * padding,
    );
    let (rect, response) = ui.allocate_exact_size(widget_size, egui::Sense::hover());
    let origin = rect.left_top() + padding_vec;

    // Define the points of the profile
    let points = [
        origin,
        origin + Vec2::new(frame_width_px, 0.0),
        origin + Vec2::new(frame_width_px, frame_depth_px),
        origin + Vec2::new(glass_frame_lip_px, frame_depth_px),
        origin + Vec2::new(glass_frame_lip_px, frame_depth_px - gap_depth_px),
        origin + Vec2::new(0.0, frame_depth_px - gap_depth_px),
        origin,
    ];

    // Render the outline
    ui.painter().add(Shape::line(
        points.to_vec(),
        Stroke::new(1.0, Color32::BLACK),
    ));

    // Render dimension lines and labels
    let text_style = egui::TextStyle::Small;
    let font_id = ui.style().text_styles[&text_style].clone();

    // Dim: Frame width
    render_dimension_line(
        ui,
        points[0],
        points[1],
        padding / 2.0,
        &format!("{:.1} {}", frame.frame_width.to_mm(), "mm"),
        &font_id,
    );

    // Dim: Frame depth
    render_dimension_line(
        ui,
        points[1],
        points[2],
        padding / 2.0,
        &format!("{:.1} {}", frame.frame_depth.to_mm(), "mm"),
        &font_id,
    );

    // Dim: gap depth
    let point_offset = Vec2::new(frame_width_px + 2.0 * padding, 0.0);
    render_dimension_line(
        ui,
        points[4] + point_offset,
        points[3] + point_offset,
        padding / 2.0,
        &format!("{:.1} {}", gap_depth_mm, "mm"),
        &font_id,
    );

    // Dim: gap width
    let point_offset = Vec2::new(0.0, gap_depth_px);
    render_dimension_line(
        ui,
        points[4] + point_offset,
        points[5] + point_offset,
        padding / 2.0,
        &format!(
            "{:.1} {}",
            frame.glass_frame_lip.to_mm() + frame.glass_frame_tol.to_mm(),
            "mm"
        ),
        &font_id,
    );

    response
}

fn render_matte_profile(
    ui: &mut egui::Ui,
    frame: &PictureFrame,
    pix_per_mm: f32,
) -> egui::Response {
    let matte_size_px = Vec2::new(
        frame.glass_size.values[0].to_mm() as f32 * pix_per_mm,
        frame.glass_size.values[1].to_mm() as f32 * pix_per_mm,
    );
    let matte_hole_size_px = Vec2::new(
        frame.matte_hole_size.values[0].to_mm() as f32 * pix_per_mm,
        frame.matte_hole_size.values[1].to_mm() as f32 * pix_per_mm,
    );

    let padding = 20.0;
    let padding_vec = Vec2::new(padding, padding);
    let widget_size = matte_size_px + 2.0 * padding_vec;
    let (rect, response) = ui.allocate_exact_size(widget_size, egui::Sense::hover());

    // Define the points of the profile
    let origin = rect.left_top() + padding_vec;
    let pts_outside = [
        origin,
        origin + Vec2::new(matte_size_px.x, 0.0),
        origin + Vec2::new(matte_size_px.x, matte_size_px.y),
        origin + Vec2::new(0.0, matte_size_px.y),
        origin,
    ];

    let border = (frame.glass_size - frame.matte_hole_size) / 2.0;
    let border_px = border.cast::<f32>().to_custom_unit(pix_per_mm);
    let origin = rect.left_top() + padding_vec + border_px.into();
    let pts_inside = [
        origin,
        origin + Vec2::new(matte_hole_size_px.x, 0.0),
        origin + Vec2::new(matte_hole_size_px.x, matte_hole_size_px.y),
        origin + Vec2::new(0.0, matte_hole_size_px.y),
        origin,
    ];

    // Render dimension lines and labels
    let text_style = egui::TextStyle::Small;
    let font_id = ui.style().text_styles[&text_style].clone();

    // Dim: matte width
    render_dimension_line(
        ui,
        pts_outside[0],
        pts_outside[1],
        padding / 2.0,
        &format!("{:.1} {}", frame.glass_size.values[0].to_mm(), "mm"),
        &font_id,
    );

    // Dim: matte height
    render_dimension_line(
        ui,
        pts_outside[1],
        pts_outside[2],
        padding / 2.0,
        &format!("{:.1} {}", frame.glass_size.values[1].to_mm(), "mm"),
        &font_id,
    );

    // Dim: matte hole width
    render_dimension_line(
        ui,
        pts_inside[0],
        pts_inside[1],
        -padding / 2.0,
        &format!("{:.1} {}", frame.matte_hole_size.values[0].to_mm(), "mm"),
        &font_id,
    );

    // Dim: matte hole height
    render_dimension_line(
        ui,
        pts_inside[1],
        pts_inside[2],
        -padding / 2.0,
        &format!("{:.1} {}", frame.matte_hole_size.values[1].to_mm(), "mm"),
        &font_id,
    );

    // Dim: top border
    render_dimension_line(
        ui,
        pts_outside[1],
        pts_outside[1] + Vec2::new(0.0, border_px[1]),
        2.0 * padding / 2.0,
        &format!("{:.1} {}", border.values[1].to_mm(), "mm"),
        &font_id,
    );

    // Dim: left border
    render_dimension_line(
        ui,
        pts_outside[3],
        pts_outside[3] + Vec2::new(border_px[0], 0.0),
        -padding / 2.0,
        &format!("{:.1} {}", border.values[0].to_mm(), "mm"),
        &font_id,
    );

    // Render the outside
    ui.painter().add(Shape::line(
        pts_outside.to_vec(),
        Stroke::new(1.0, Color32::BLACK),
    ));

    // Render the inside
    ui.painter().add(Shape::line(
        pts_inside.to_vec(),
        Stroke::new(1.0, Color32::BLACK),
    ));

    response
}

// Add this helper function outside of the impl block
fn standard_size_menu_button(ui: &mut egui::Ui, size: &mut LengthVec<f64, 2>) {
    ui.menu_button("📏", |ui| {
        for (label, width, height) in [
            ("4x6\"", 4.0, 6.0),
            ("5x7\"", 5.0, 7.0),
            ("8x10\"", 8.0, 10.0),
            ("11x14\"", 11.0, 14.0),
        ] {
            if ui.button(label).clicked() {
                *size = LengthVec::new([width, height], LengthUnit::InchesFractional);
                ui.close_menu();
            }
        }
    });
}

#[cfg(not(target_arch = "wasm32"))]
fn execute<F: Future<Output = ()> + Send + 'static>(f: F) {
    // this is stupid... use any executor of your choice instead
    std::thread::spawn(move || futures::executor::block_on(f));
}

#[cfg(target_arch = "wasm32")]
fn execute<F: Future<Output = ()> + 'static>(f: F) {
    wasm_bindgen_futures::spawn_local(f);
}
