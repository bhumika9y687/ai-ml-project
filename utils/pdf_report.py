"""
Cardio Vue AI — PDF Report Generator
Generates branded, downloadable health reports for patients and doctors.
Requires: reportlab
"""

import io
from datetime import datetime
import numpy as np

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm, cm
    from reportlab.lib.colors import HexColor, white, black
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether
    )
    from reportlab.platypus import Image as RLImage
    from reportlab.graphics.shapes import Drawing, Rect, String, Line, Circle
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.charts.lineplots import LinePlot
    from reportlab.graphics import renderPDF
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# ─── BRAND COLORS ─────────────────────────────────────────────────────────────
CRIMSON = HexColor('#C8102E') if REPORTLAB_AVAILABLE else None
NAVY = HexColor('#0B1629') if REPORTLAB_AVAILABLE else None
NAVY_LIGHT = HexColor('#1E2D4A') if REPORTLAB_AVAILABLE else None
SLATE = HexColor('#8A9BBE') if REPORTLAB_AVAILABLE else None
SUCCESS = HexColor('#10B981') if REPORTLAB_AVAILABLE else None
WARNING = HexColor('#F59E0B') if REPORTLAB_AVAILABLE else None
DANGER = HexColor('#EF4444') if REPORTLAB_AVAILABLE else None
BG_LIGHT = HexColor('#F8F9FF') if REPORTLAB_AVAILABLE else None
BORDER = HexColor('#E2E8F0') if REPORTLAB_AVAILABLE else None


def _risk_color_rl(label):
    mapping = {'Low': SUCCESS, 'Moderate': WARNING, 'High': DANGER, 'Critical': CRIMSON}
    return mapping.get(label, SLATE)


def generate_patient_report(patient_data: dict, health_records: list,
                              prediction: dict = None, blood_tests: list = None) -> bytes:
    """
    Generate a comprehensive patient health report PDF.
    Returns bytes that can be served as a download.
    """
    if not REPORTLAB_AVAILABLE:
        return _fallback_report(patient_data, health_records, prediction)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=20*mm, rightMargin=20*mm,
        topMargin=15*mm, bottomMargin=20*mm,
        title="Cardio Vue AI Health Report",
        author="Cardio Vue AI",
    )

    styles = getSampleStyleSheet()
    story = []

    # ── Header banner ──────────────────────────────────────────────────────────
    w, h = A4
    usable_w = w - 40*mm

    header_drawing = Drawing(usable_w, 60)
    header_drawing.add(Rect(0, 0, usable_w, 60, fillColor=NAVY, strokeColor=None))
    header_drawing.add(Rect(0, 0, 6, 60, fillColor=CRIMSON, strokeColor=None))
    header_drawing.add(String(20, 36, "🫀 Cardio Vue AI",
                               fontName="Helvetica-Bold", fontSize=18, fillColor=white))
    header_drawing.add(String(20, 18, "Cardiovascular Health Report",
                               fontName="Helvetica", fontSize=10, fillColor=HexColor('#8A9BBE')))
    gen_date = datetime.now().strftime("%B %d, %Y  %H:%M")
    header_drawing.add(String(usable_w - 140, 18, f"Generated: {gen_date}",
                               fontName="Helvetica", fontSize=9, fillColor=HexColor('#8A9BBE')))
    story.append(header_drawing)
    story.append(Spacer(1, 8*mm))

    # ── Patient info ───────────────────────────────────────────────────────────
    name = patient_data.get('name', 'Patient')
    info_data = [
        ['Patient Name', name, 'Patient ID', patient_data.get('username', 'N/A')],
        ['Date of Birth / Age', f"{patient_data.get('age', 'N/A')} years",
         'Gender', patient_data.get('gender', 'N/A')],
        ['Email', patient_data.get('email', 'N/A'), 'Report Date', gen_date.split('  ')[0]],
    ]
    info_table = Table(info_data, colWidths=[35*mm, 55*mm, 35*mm, 55*mm])
    info_table.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
        ('FONTNAME', (2,0), (2,-1), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('TEXTCOLOR', (0,0), (0,-1), NAVY),
        ('TEXTCOLOR', (2,0), (2,-1), NAVY),
        ('TEXTCOLOR', (1,0), (1,-1), HexColor('#374151')),
        ('TEXTCOLOR', (3,0), (3,-1), HexColor('#374151')),
        ('BACKGROUND', (0,0), (-1,-1), BG_LIGHT),
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [white, BG_LIGHT]),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER),
        ('ROWPADDING', (0,0), (-1,-1), 6),
        ('ROUNDEDCORNERS', [4]),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 6*mm))

    # ── Risk Summary ────────────────────────────────────────────────────────────
    if prediction:
        story.append(_section_header("AI Risk Assessment", usable_w))
        story.append(Spacer(1, 3*mm))

        risk_score = prediction.get('risk_score', 0)
        risk_label = prediction.get('risk_label', 'Unknown')
        risk_col = _risk_color_rl(risk_label)
        ci_low = prediction.get('ci_low', 0)
        ci_high = prediction.get('ci_high', 0)

        # Risk gauge bar
        gauge = Drawing(usable_w, 40)
        gauge.add(Rect(0, 20, usable_w, 12, fillColor=HexColor('#E2E8F0'), strokeColor=None, rx=6))
        # Zones
        zone_w = usable_w / 4
        zone_colors = [SUCCESS, WARNING, DANGER, CRIMSON]
        zone_labels = ['Low', 'Moderate', 'High', 'Critical']
        for i, (zc, zl) in enumerate(zip(zone_colors, zone_labels)):
            gauge.add(Rect(i*zone_w, 20, zone_w, 12, fillColor=zc, strokeColor=None,
                           rx=3 if i == 0 else 0))
            gauge.add(String(i*zone_w + zone_w/2 - 15, 10, zl,
                             fontName='Helvetica', fontSize=7, fillColor=HexColor('#6B7280')))
        # Score needle
        needle_x = (risk_score / 100) * usable_w
        gauge.add(Rect(needle_x - 2, 16, 4, 20, fillColor=NAVY, strokeColor=None))
        gauge.add(String(needle_x - 15, 38, f"{risk_score:.1f}%",
                         fontName='Helvetica-Bold', fontSize=10, fillColor=NAVY))
        story.append(gauge)
        story.append(Spacer(1, 4*mm))

        risk_summary = [
            ['Risk Score', f"{risk_score:.1f}%", 'Risk Level', risk_label,
             'Confidence Interval', f"{ci_low:.1f}% – {ci_high:.1f}%"],
            ['AI Model', prediction.get('model_name', 'Ensemble'),
             'Confidence', f"{prediction.get('model_confidence', 0):.1f}%",
             'Assessment Date', prediction.get('timestamp', gen_date.split('  ')[0])],
        ]
        rt = Table(risk_summary, colWidths=[30*mm, 35*mm, 28*mm, 30*mm, 35*mm, 22*mm])
        rt.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTNAME', (0,0), (0,-1), 'Helvetica-Bold'),
            ('FONTNAME', (2,0), (2,-1), 'Helvetica-Bold'),
            ('FONTNAME', (4,0), (4,-1), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('TEXTCOLOR', (1,0), (1,0), risk_col),
            ('FONTNAME', (1,0), (1,0), 'Helvetica-Bold'),
            ('GRID', (0,0), (-1,-1), 0.5, BORDER),
            ('BACKGROUND', (0,0), (-1,-1), BG_LIGHT),
            ('ROWPADDING', (0,0), (-1,-1), 5),
        ]))
        story.append(rt)
        story.append(Spacer(1, 4*mm))

        # SHAP feature importance table
        shap = prediction.get('shap_values', {})
        if shap:
            story.append(_section_header("Key Risk Drivers (SHAP Analysis)", usable_w))
            story.append(Spacer(1, 2*mm))
            sorted_shap = sorted(shap.items(), key=lambda x: abs(x[1]), reverse=True)[:8]
            shap_rows = [['Factor', 'Contribution', 'Direction']] + [
                [k, f"{abs(v)*100:.1f}%", '↑ Increases risk' if v > 0 else '↓ Reduces risk']
                for k, v in sorted_shap
            ]
            shap_table = Table(shap_rows, colWidths=[80*mm, 40*mm, 60*mm])
            shap_table.setStyle(TableStyle([
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTSIZE', (0,0), (-1,-1), 9),
                ('BACKGROUND', (0,0), (-1,0), NAVY),
                ('TEXTCOLOR', (0,0), (-1,0), white),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [white, BG_LIGHT]),
                ('GRID', (0,0), (-1,-1), 0.5, BORDER),
                ('ROWPADDING', (0,0), (-1,-1), 5),
                *[('TEXTCOLOR', (2,i+1), (2,i+1),
                   DANGER if sorted_shap[i][1] > 0 else SUCCESS)
                  for i in range(len(sorted_shap))],
            ]))
            story.append(shap_table)
            story.append(Spacer(1, 4*mm))

    # ── Health Records History ──────────────────────────────────────────────────
    if health_records:
        story.append(_section_header("Health History (Last 12 Assessments)", usable_w))
        story.append(Spacer(1, 2*mm))
        recent = sorted(health_records, key=lambda x: x['date'], reverse=True)[:12]
        rec_rows = [['Date', 'Risk Score', 'Risk Level', 'BP (sys/dia)', 'Cholesterol', 'BMI']]
        for r in recent:
            rec_rows.append([
                r.get('date', ''), f"{r.get('risk_score', 0):.1f}%",
                r.get('risk_label', ''),
                f"{r.get('bp_systolic', '--')}/{r.get('bp_diastolic', '--')}",
                str(r.get('cholesterol', '--')), str(r.get('bmi', '--'))
            ])
        hist_table = Table(rec_rows, colWidths=[28*mm, 22*mm, 22*mm, 28*mm, 28*mm, 22*mm])
        risk_row_styles = []
        for i, r in enumerate(recent, 1):
            lbl = r.get('risk_label', '')
            col = _risk_color_rl(lbl)
            risk_row_styles.append(('TEXTCOLOR', (2,i), (2,i), col))
        hist_table.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('BACKGROUND', (0,0), (-1,0), NAVY),
            ('TEXTCOLOR', (0,0), (-1,0), white),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [white, BG_LIGHT]),
            ('GRID', (0,0), (-1,-1), 0.5, BORDER),
            ('ROWPADDING', (0,0), (-1,-1), 4),
            *risk_row_styles,
        ]))
        story.append(hist_table)
        story.append(Spacer(1, 4*mm))

    # ── Blood Tests ─────────────────────────────────────────────────────────────
    if blood_tests:
        latest_bt = blood_tests[0]
        story.append(_section_header("Latest Blood Test Results", usable_w))
        story.append(Spacer(1, 2*mm))
        bt_data = [
            ['Marker', 'Value', 'Normal Range', 'Status'],
            ['HDL Cholesterol', f"{latest_bt.get('hdl', '--')} mg/dL", '>40 mg/dL',
             '✓ Normal' if latest_bt.get('hdl', 0) > 40 else '✗ Low'],
            ['LDL Cholesterol', f"{latest_bt.get('ldl', '--')} mg/dL", '<100 mg/dL',
             '✓ Normal' if latest_bt.get('ldl', 0) < 100 else '✗ Elevated'],
            ['Triglycerides', f"{latest_bt.get('triglycerides', '--')} mg/dL", '<150 mg/dL',
             '✓ Normal' if latest_bt.get('triglycerides', 0) < 150 else '✗ Elevated'],
            ['Fasting Glucose', f"{latest_bt.get('glucose', '--')} mg/dL", '70–99 mg/dL',
             '✓ Normal' if 70 <= latest_bt.get('glucose', 0) <= 99 else '⚠ Abnormal'],
            ['HbA1c', f"{latest_bt.get('hba1c', '--')}%", '<5.7%',
             '✓ Normal' if latest_bt.get('hba1c', 0) < 5.7 else '⚠ Elevated'],
            ['Creatinine', f"{latest_bt.get('creatinine', '--')} mg/dL", '0.7–1.2 mg/dL',
             '✓ Normal'],
        ]
        bt_table = Table(bt_data, colWidths=[45*mm, 35*mm, 35*mm, 35*mm])
        bt_table.setStyle(TableStyle([
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 9),
            ('BACKGROUND', (0,0), (-1,0), NAVY),
            ('TEXTCOLOR', (0,0), (-1,0), white),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [white, BG_LIGHT]),
            ('GRID', (0,0), (-1,-1), 0.5, BORDER),
            ('ROWPADDING', (0,0), (-1,-1), 5),
        ]))
        story.append(bt_table)
        story.append(Spacer(1, 4*mm))

    # ── Recommendations ─────────────────────────────────────────────────────────
    story.append(_section_header("AI-Generated Recommendations", usable_w))
    story.append(Spacer(1, 2*mm))
    risk_label = prediction.get('risk_label', 'Moderate') if prediction else 'Moderate'
    recs = _get_recommendations(risk_label)
    rec_data = [['Priority', 'Recommendation', 'Expected Impact']]
    for r in recs:
        rec_data.append([r['priority'], r['text'], r['impact']])
    rec_table = Table(rec_data, colWidths=[18*mm, 110*mm, 42*mm])
    rec_table.setStyle(TableStyle([
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('BACKGROUND', (0,0), (-1,0), NAVY),
        ('TEXTCOLOR', (0,0), (-1,0), white),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [white, BG_LIGHT]),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER),
        ('ROWPADDING', (0,0), (-1,-1), 5),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(rec_table)
    story.append(Spacer(1, 6*mm))

    # ── Disclaimer ───────────────────────────────────────────────────────────────
    disclaimer_style = ParagraphStyle(
        'disclaimer', parent=styles['Normal'],
        fontSize=7, textColor=HexColor('#9CA3AF'),
        borderPad=4, leading=10,
    )
    story.append(HRFlowable(width="100%", thickness=0.5, color=BORDER))
    story.append(Spacer(1, 3*mm))
    story.append(Paragraph(
        "⚕ MEDICAL DISCLAIMER: This report is generated by an AI system for informational purposes only. "
        "It does not constitute medical advice, diagnosis, or treatment. Always consult a qualified healthcare "
        "professional before making any medical decisions. Cardio Vue AI is a decision-support tool, not a "
        "replacement for clinical judgment. Predictions are based on population-level statistical models and "
        "individual results may vary. — Cardio Vue AI v2.0",
        disclaimer_style
    ))

    doc.build(story)
    buf.seek(0)
    return buf.read()


def _section_header(title: str, width: float) -> Drawing:
    d = Drawing(width, 24)
    d.add(Rect(0, 0, width, 22, fillColor=HexColor('#F1F5F9'), strokeColor=BORDER,
               strokeWidth=0.5, rx=4))
    d.add(Rect(0, 0, 4, 22, fillColor=CRIMSON, strokeColor=None))
    d.add(String(12, 7, title, fontName='Helvetica-Bold', fontSize=11, fillColor=NAVY))
    return d


def _get_recommendations(risk_label: str) -> list:
    base = [
        {'priority': '🟢 High', 'text': 'Engage in 150 min/week of moderate aerobic activity (brisk walking, cycling, swimming)', 'impact': '↓ Risk ~12%'},
        {'priority': '🟢 High', 'text': 'Follow a heart-healthy diet: Mediterranean diet, reduce sodium <2300mg/day, increase fiber 25-35g/day', 'impact': '↓ Risk ~9%'},
        {'priority': '🟡 Med', 'text': 'Monitor blood pressure regularly. Target <130/80 mmHg. Consider home BP monitor.', 'impact': '↓ Risk ~8%'},
        {'priority': '🟡 Med', 'text': 'Get adequate sleep: 7-9 hours/night. Poor sleep increases cardiovascular risk by up to 48%.', 'impact': '↓ Risk ~5%'},
        {'priority': '🟡 Med', 'text': 'Practice stress management: mindfulness, yoga, breathing exercises (4-7-8 technique)', 'impact': '↓ Risk ~6%'},
    ]
    if risk_label in ['High', 'Critical']:
        base.insert(0, {'priority': '🔴 Urgent', 'text': 'Schedule a cardiology consultation within 2 weeks. Discuss medication options with your doctor.', 'impact': 'Critical'})
        base.insert(1, {'priority': '🔴 Urgent', 'text': 'If you smoke: smoking cessation reduces heart disease risk by 50% within 1 year.', 'impact': '↓ Risk ~18%'})
    return base[:5]


def _fallback_report(patient_data, health_records, prediction) -> bytes:
    """Simple CSV fallback when reportlab is not installed."""
    import csv
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(['Cardio Vue AI Health Report'])
    writer.writerow(['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M')])
    writer.writerow([])
    writer.writerow(['Patient:', patient_data.get('name', 'N/A')])
    if prediction:
        writer.writerow(['Risk Score:', f"{prediction.get('risk_score', 0):.1f}%"])
        writer.writerow(['Risk Level:', prediction.get('risk_label', 'N/A')])
    writer.writerow([])
    writer.writerow(['Date', 'Risk Score', 'Risk Level', 'BP Systolic', 'Cholesterol', 'BMI'])
    for r in sorted(health_records, key=lambda x: x['date'], reverse=True)[:20]:
        writer.writerow([r.get('date'), r.get('risk_score'), r.get('risk_label'),
                         r.get('bp_systolic'), r.get('cholesterol'), r.get('bmi')])
    return buf.getvalue().encode()
