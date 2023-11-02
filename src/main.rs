#![warn(clippy::all, rust_2018_idioms)]
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use picture_frame_calc::PictureFrameApp;

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result<()> {
    tracing_subscriber::fmt::init();
    let native_options = eframe::NativeOptions::default();
    eframe::run_native(
        "Picture Frame",
        native_options,
        Box::new(|cc| Ok(Box::new(PictureFrameApp::new(cc)))),
    )
}

#[cfg(target_arch = "wasm32")]
fn main() {
    // Redirect `log` message to `console.log` and friends:
    eframe::WebLogger::init(log::LevelFilter::Debug).ok();

    let web_options = eframe::WebOptions::default();

    wasm_bindgen_futures::spawn_local(async {
        eframe::WebRunner::new()
            .start(
                "egui_canvas",
                web_options,
                Box::new(|cc| Ok(Box::new(PictureFrameApp::new(cc)))),
            )
            .await
            .expect("failed to start eframe");
    });
}
