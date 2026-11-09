"""
Tamil Nadu Agricultural Decision Support System
Standalone Desktop Application for Farmers

Framework: CustomTkinter (Desktop GUI) + PyTorch FT-Transformer (XAI)
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

# Ensure proper path resolution for development and PyInstaller standalone bundle
def get_base_dir():
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return sys._MEIPASS
    return os.path.dirname(os.path.abspath(__file__))

BASE_DIR = get_base_dir()
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.farmer_xai import FarmerFriendlyAdvisor

# Initialize CustomTkinter appearance
ctk.set_appearance_mode("Light")
ctk.set_default_color_theme("green")


class FarmerApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Tamil Nadu Crop Advisory & Decision Support System")
        self.geometry("1180x820")
        self.minsize(1000, 700)

        # Load Advisor Engine
        models_dir = os.path.join(BASE_DIR, "models")
        self.advisor = FarmerFriendlyAdvisor(models_dir)

        # Soil Options available in dataset
        self.soil_options = [
            "Alluvial soil", "Black soil", "Black cotton soil", "Clay soil",
            "Clay loamy soil", "Laterite soil", "Loamy soil", "Red soil",
            "Red lateritic loamy soil", "Sandy soil", "Sandy loamy soil",
            "Silty loamy soil", "Well-drained soil"
        ]

        # Current Advice cache
        self.current_advice = None
        self.current_farmer_input = None

        self._build_ui()

    def _build_ui(self):
        # -------------------------------------------------------------
        # TOP HEADER
        # -------------------------------------------------------------
        header_frame = ctk.CTkFrame(self, fg_color="#1b4332", corner_radius=0, height=85)
        header_frame.pack(fill="x", side="top")
        header_frame.pack_propagate(False)

        title_label = ctk.CTkLabel(
            header_frame,
            text="🌾 TAMIL NADU CROP ADVISORY & SOIL HEALTH SYSTEM",
            font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
            text_color="#ffffff"
        )
        title_label.pack(side="left", padx=25, pady=(15, 2))

        subtitle_label = ctk.CTkLabel(
            header_frame,
            text="AI-Guided Crop Selection & Fertilizer Advisory for Farmers across Tamil Nadu Agro-Climatic Zones",
            font=ctk.CTkFont(family="Segoe UI", size=13),
            text_color="#b7dfca"
        )
        subtitle_label.place(x=25, y=48)

        # -------------------------------------------------------------
        # MAIN DUAL-PANE CONTAINER
        # -------------------------------------------------------------
        main_pane = ctk.CTkFrame(self, fg_color="#f3f4f6")
        main_pane.pack(fill="both", expand=True, padx=15, pady=15)

        # LEFT PANE: INPUT CONTROLS (Scrollable)
        self.left_pane = ctk.CTkScrollableFrame(
            main_pane, width=460, fg_color="#ffffff", corner_radius=10,
            scrollbar_button_color="#2d6a4f", scrollbar_button_hover_color="#1b4332"
        )
        self.left_pane.pack(side="left", fill="both", padx=(0, 10), pady=0)

        # RIGHT PANE: VISUAL RECOMMENDATION & REASONING (Scrollable)
        self.right_pane = ctk.CTkScrollableFrame(
            main_pane, fg_color="#ffffff", corner_radius=10,
            scrollbar_button_color="#2d6a4f", scrollbar_button_hover_color="#1b4332"
        )
        self.right_pane.pack(side="right", fill="both", expand=True, padx=0, pady=0)

        self._build_input_controls()
        self._build_placeholder_results()

    def _build_input_controls(self):
        """Construct clean, accessible inputs with large text and real-time slider value labels."""
        lp = self.left_pane

        # SECTION 1: FIELD & SOIL ENVIRONMENT
        sec1_title = ctk.CTkLabel(
            lp, text="📍 1. Land & Cropping Conditions",
            font=ctk.CTkFont(size=16, weight="bold"), text_color="#1b4332"
        )
        sec1_title.pack(anchor="w", padx=15, pady=(15, 10))

        # Soil Type Dropdown
        ctk.CTkLabel(lp, text="Soil Texture / Type:", font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=15, pady=(5, 2))
        self.soil_var = ctk.StringVar(value="Alluvial soil")
        self.soil_menu = ctk.CTkOptionMenu(
            lp, values=self.soil_options, variable=self.soil_var,
            fg_color="#2d6a4f", button_color="#1b4332", font=ctk.CTkFont(size=13), height=38
        )
        self.soil_menu.pack(fill="x", padx=15, pady=(0, 12))

        # Cropping Season Dropdown
        ctk.CTkLabel(lp, text="Cropping Season:", font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=15, pady=(5, 2))
        self.season_var = ctk.StringVar(value="Kharif (Monsoon / ஆடிப் பட்டம்)")
        self.season_menu = ctk.CTkOptionMenu(
            lp,
            values=[
                "Kharif (Monsoon / ஆடிப் பட்டம்)",
                "Rabi (Winter / கார்த்திகைப் பட்டம்)",
                "Zaid (Summer / சித்திரைப் பட்டம்)"
            ],
            variable=self.season_var,
            fg_color="#2d6a4f", button_color="#1b4332", font=ctk.CTkFont(size=13), height=38
        )
        self.season_menu.pack(fill="x", padx=15, pady=(0, 12))

        # Water Source Segmented Button
        ctk.CTkLabel(lp, text="Water Availability Mode:", font=ctk.CTkFont(size=13, weight="bold")).pack(anchor="w", padx=15, pady=(5, 2))
        self.water_seg = ctk.CTkSegmentedButton(
            lp, values=["Irrigated (பாசனம்)", "Rainfed (மானாவாரி)"],
            selected_color="#2d6a4f", selected_hover_color="#1b4332", height=36
        )
        self.water_seg.set("Irrigated (பாசனம்)")
        self.water_seg.pack(fill="x", padx=15, pady=(0, 20))

        # SECTION 2: SOIL HEALTH CARD (NUTRIENTS)
        sec2_title = ctk.CTkLabel(
            lp, text="🧪 2. Soil Health Card Results",
            font=ctk.CTkFont(size=16, weight="bold"), text_color="#1b4332"
        )
        sec2_title.pack(anchor="w", padx=15, pady=(10, 10))

        # Nitrogen (N) Slider
        self.n_label = ctk.CTkLabel(lp, text="Nitrogen (N): 85 kg/ha", font=ctk.CTkFont(size=13, weight="bold"))
        self.n_label.pack(anchor="w", padx=15, pady=(2, 2))
        self.n_slider = ctk.CTkSlider(lp, from_=10, to=200, number_of_steps=190, command=self._update_n_label)
        self.n_slider.set(85)
        self.n_slider.pack(fill="x", padx=15, pady=(0, 12))

        # Phosphorus (P) Slider
        self.p_label = ctk.CTkLabel(lp, text="Phosphorus (P): 45 kg/ha", font=ctk.CTkFont(size=13, weight="bold"))
        self.p_label.pack(anchor="w", padx=15, pady=(2, 2))
        self.p_slider = ctk.CTkSlider(lp, from_=5, to=120, number_of_steps=115, command=self._update_p_label)
        self.p_slider.set(45)
        self.p_slider.pack(fill="x", padx=15, pady=(0, 12))

        # Potassium (K) Slider
        self.k_label = ctk.CTkLabel(lp, text="Potassium (K): 45 kg/ha", font=ctk.CTkFont(size=13, weight="bold"))
        self.k_label.pack(anchor="w", padx=15, pady=(2, 2))
        self.k_slider = ctk.CTkSlider(lp, from_=5, to=150, number_of_steps=145, command=self._update_k_label)
        self.k_slider.set(45)
        self.k_slider.pack(fill="x", padx=15, pady=(0, 12))

        # Soil pH Slider
        self.ph_label = ctk.CTkLabel(lp, text="Soil Reaction (pH): 6.8", font=ctk.CTkFont(size=13, weight="bold"))
        self.ph_label.pack(anchor="w", padx=15, pady=(2, 2))
        self.ph_slider = ctk.CTkSlider(lp, from_=4.0, to=9.5, number_of_steps=55, command=self._update_ph_label)
        self.ph_slider.set(6.8)
        self.ph_slider.pack(fill="x", padx=15, pady=(0, 20))

        # SECTION 3: LOCAL CLIMATE
        sec3_title = ctk.CTkLabel(
            lp, text="🌤️ 3. Prevailing Local Weather",
            font=ctk.CTkFont(size=16, weight="bold"), text_color="#1b4332"
        )
        sec3_title.pack(anchor="w", padx=15, pady=(10, 10))

        # Temperature Slider
        self.temp_label = ctk.CTkLabel(lp, text="Temperature: 27.0 °C", font=ctk.CTkFont(size=13, weight="bold"))
        self.temp_label.pack(anchor="w", padx=15, pady=(2, 2))
        self.temp_slider = ctk.CTkSlider(lp, from_=15.0, to=45.0, number_of_steps=60, command=self._update_temp_label)
        self.temp_slider.set(27.0)
        self.temp_slider.pack(fill="x", padx=15, pady=(0, 12))

        # Humidity Slider
        self.hum_label = ctk.CTkLabel(lp, text="Relative Humidity: 78 %", font=ctk.CTkFont(size=13, weight="bold"))
        self.hum_label.pack(anchor="w", padx=15, pady=(2, 2))
        self.hum_slider = ctk.CTkSlider(lp, from_=20, to=100, number_of_steps=80, command=self._update_hum_label)
        self.hum_slider.set(78)
        self.hum_slider.pack(fill="x", padx=15, pady=(0, 20))

        # BIG ACTION BUTTON
        self.recommend_btn = ctk.CTkButton(
            lp, text="🌾 GET CROP RECOMMENDATION",
            font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold"),
            fg_color="#15803d", hover_color="#166534",
            height=46, corner_radius=8,
            command=self._on_recommend_clicked
        )
        self.recommend_btn.pack(fill="x", padx=15, pady=(10, 25))

    # Real-time label updates for sliders
    def _update_n_label(self, val):
        self.n_label.configure(text=f"Nitrogen (N): {int(val)} kg/ha")

    def _update_p_label(self, val):
        self.p_label.configure(text=f"Phosphorus (P): {int(val)} kg/ha")

    def _update_k_label(self, val):
        self.k_label.configure(text=f"Potassium (K): {int(val)} kg/ha")

    def _update_ph_label(self, val):
        self.ph_label.configure(text=f"Soil Reaction (pH): {val:.1f}")

    def _update_temp_label(self, val):
        self.temp_label.configure(text=f"Temperature: {val:.1f} °C")

    def _update_hum_label(self, val):
        self.hum_label.configure(text=f"Relative Humidity: {int(val)} %")

    def _build_placeholder_results(self):
        """Initial state before user clicks recommend."""
        rp = self.right_pane
        for w in rp.winfo_children():
            w.destroy()

        placeholder = ctk.CTkFrame(rp, fg_color="#f8faf9", corner_radius=10)
        placeholder.pack(fill="both", expand=True, padx=20, pady=30)

        icon_lbl = ctk.CTkLabel(placeholder, text="🌾", font=ctk.CTkFont(size=64))
        icon_lbl.pack(pady=(60, 10))

        welcome_lbl = ctk.CTkLabel(
            placeholder,
            text="Ready to Analyze Your Farm Conditions",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="bold"),
            text_color="#1b4332"
        )
        welcome_lbl.pack(pady=(0, 10))

        desc_lbl = ctk.CTkLabel(
            placeholder,
            text="1. Adjust your soil type and nutrient levels on the left panel.\n"
                 "2. Set current season and expected weather conditions.\n"
                 "3. Click 'GET CROP RECOMMENDATION' to view the optimal crop,\n"
                 "   plain-language reasons, and actionable soil advice.",
            font=ctk.CTkFont(size=14),
            text_color="#4b5563",
            justify="center"
        )
        desc_lbl.pack(pady=(0, 40))

    def _on_recommend_clicked(self):
        # Extract inputs
        soil_raw = self.soil_var.get().strip().lower()
        season_raw = "kharif" if "kharif" in self.season_var.get().lower() else ("rabi" if "rabi" in self.season_var.get().lower() else "zaid")
        water_raw = "irrigated" if "irrigated" in self.water_seg.get().lower() else "rainfed"

        farmer_input = {
            'N': float(self.n_slider.get()),
            'P': float(self.p_slider.get()),
            'K': float(self.k_slider.get()),
            'SOIL_PH': float(self.ph_slider.get()),
            'TEMP': float(self.temp_slider.get()),
            'RELATIVE_HUMIDITY': float(self.hum_slider.get()),
            'SOIL': soil_raw,
            'SEASON': season_raw,
            'WATER_SOURCE': water_raw
        }
        self.current_farmer_input = farmer_input

        # Compute advice via FarmerFriendlyAdvisor
        advice = self.advisor.advise(farmer_input)
        self.current_advice = advice

        # Render visual recommendation view
        self._render_results(advice, farmer_input)

    def _render_results(self, advice: dict, farmer_input: dict):
        rp = self.right_pane
        for w in rp.winfo_children():
            w.destroy()

        # -------------------------------------------------------------
        # 1. PRIMARY RECOMMENDATION CARD
        # -------------------------------------------------------------
        hero_card = ctk.CTkFrame(rp, fg_color="#f0fdf4", border_color="#86efac", border_width=2, corner_radius=12)
        hero_card.pack(fill="x", padx=15, pady=(15, 15))

        top_row = ctk.CTkFrame(hero_card, fg_color="transparent")
        top_row.pack(fill="x", padx=20, pady=(15, 5))

        badge = ctk.CTkLabel(
            top_row, text=f"  {advice['badge_text']}  ",
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=advice["badge_color"], text_color="#ffffff", corner_radius=6
        )
        badge.pack(side="left")

        crop_icon = ctk.CTkLabel(hero_card, text=advice["icon"], font=ctk.CTkFont(size=44))
        crop_icon.pack(anchor="w", padx=20, pady=(5, 0))

        crop_title = ctk.CTkLabel(
            hero_card,
            text=f"RECOMMENDED CROP: {advice['crop_name'].upper()}",
            font=ctk.CTkFont(family="Segoe UI", size=24, weight="bold"),
            text_color="#14532d"
        )
        crop_title.pack(anchor="w", padx=20, pady=(0, 4))

        meta_text = f"Category: {advice['category']}   |   Water Regime: {advice['water_need']}   |   Cultivation: {advice['growth_type']}"
        meta_lbl = ctk.CTkLabel(
            hero_card, text=meta_text,
            font=ctk.CTkFont(size=13), text_color="#166534"
        )
        meta_lbl.pack(anchor="w", padx=20, pady=(0, 15))

        # -------------------------------------------------------------
        # 2. WHY THIS CROP IS BEST FOR YOUR LAND? (PLAIN-LANGUAGE REASONS)
        # -------------------------------------------------------------
        reason_frame = ctk.CTkFrame(rp, fg_color="#ffffff", border_color="#e5e7eb", border_width=1, corner_radius=10)
        reason_frame.pack(fill="x", padx=15, pady=(0, 15))

        r_title = ctk.CTkLabel(
            reason_frame, text="💡 Why is this crop the best choice for your land?",
            font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"), text_color="#1f2937"
        )
        r_title.pack(anchor="w", padx=15, pady=(15, 10))

        for item in advice["reasons"]:
            card_item = ctk.CTkFrame(reason_frame, fg_color="#f8faf9", corner_radius=8, border_color="#d1fae5", border_width=1)
            card_item.pack(fill="x", padx=15, pady=(0, 10))

            it_title = ctk.CTkLabel(
                card_item, text=item["title"],
                font=ctk.CTkFont(size=14, weight="bold"), text_color="#065f46"
            )
            it_title.pack(anchor="w", padx=12, pady=(8, 2))

            it_desc = ctk.CTkLabel(
                card_item, text=item["text"],
                font=ctk.CTkFont(size=13), text_color="#374151",
                wraplength=520, justify="left"
            )
            it_desc.pack(anchor="w", padx=12, pady=(0, 10))

        # -------------------------------------------------------------
        # 3. SECONDARY ALTERNATIVES
        # -------------------------------------------------------------
        if advice["alternatives"]:
            alt_frame = ctk.CTkFrame(rp, fg_color="#f9fafb", border_color="#e5e7eb", border_width=1, corner_radius=10)
            alt_frame.pack(fill="x", padx=15, pady=(0, 15))

            alt_header = ctk.CTkLabel(
                alt_frame, text="🌱 Secondary Alternate Choices for this Plot",
                font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold"), text_color="#374151"
            )
            alt_header.pack(anchor="w", padx=15, pady=(12, 8))

            alt_cols = ctk.CTkFrame(alt_frame, fg_color="transparent")
            alt_cols.pack(fill="x", padx=15, pady=(0, 12))

            for cand in advice["alternatives"]:
                cand_box = ctk.CTkFrame(alt_cols, fg_color="#ffffff", border_color="#d1d5db", border_width=1, corner_radius=8)
                cand_box.pack(side="left", fill="x", expand=True, padx=4)

                c_name = ctk.CTkLabel(
                    cand_box, text=f"{cand['icon']} {cand['crop']}",
                    font=ctk.CTkFont(size=14, weight="bold"), text_color="#111827"
                )
                c_name.pack(anchor="w", padx=10, pady=(8, 2))

                c_pct = ctk.CTkLabel(
                    cand_box, text=f"{cand['category']} — {cand['probability_pct']}% Match",
                    font=ctk.CTkFont(size=12), text_color="#4b5563"
                )
                c_pct.pack(anchor="w", padx=10, pady=(0, 8))

        # -------------------------------------------------------------
        # 4. ACTIONABLE SOIL HEALTH & FERTILIZER SCHEDULE
        # -------------------------------------------------------------
        fert_frame = ctk.CTkFrame(rp, fg_color="#fffbeb", border_color="#fde68a", border_width=1, corner_radius=10)
        fert_frame.pack(fill="x", padx=15, pady=(0, 15))

        f_title = ctk.CTkLabel(
            fert_frame, text="🚜 Actionable Soil Health & Fertilizer Guidance",
            font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold"), text_color="#92400e"
        )
        f_title.pack(anchor="w", padx=15, pady=(12, 6))

        for tip in advice["fertilizer_tips"]:
            t_lbl = ctk.CTkLabel(
                fert_frame, text=f"• {tip}",
                font=ctk.CTkFont(size=13), text_color="#78350f",
                wraplength=520, justify="left"
            )
            t_lbl.pack(anchor="w", padx=15, pady=(2, 4))

        # -------------------------------------------------------------
        # 5. EXPORT ADVISORY BUTTON
        # -------------------------------------------------------------
        btn_frame = ctk.CTkFrame(rp, fg_color="transparent")
        btn_frame.pack(fill="x", padx=15, pady=(5, 20))

        export_btn = ctk.CTkButton(
            btn_frame, text="💾 Save / Print Farmer Advisory Card (.txt)",
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color="#1e40af", hover_color="#1e3a8a", height=42, corner_radius=8,
            command=self._export_advisory
        )
        export_btn.pack(side="left", padx=(0, 10))

    def _export_advisory(self):
        if not self.current_advice or not self.current_farmer_input:
            messagebox.showwarning("No Data", "Please generate a recommendation first.")
            return

        report_text = self.advisor.generate_farmer_report_text(self.current_farmer_input, self.current_advice)

        default_filename = f"Crop_Advisory_{self.current_advice['crop_name'].replace(' ', '_')}.txt"
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            initialfile=default_filename,
            filetypes=[("Text Documents", "*.txt"), ("All Files", "*.*")]
        )

        if file_path:
            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(report_text)
                messagebox.showinfo("Saved Successfully", f"Advisory report saved to:\n{file_path}")
            except Exception as e:
                messagebox.showerror("Save Error", f"Failed to save file:\n{str(e)}")


def main():
    app = FarmerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
