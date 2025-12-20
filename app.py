from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS

import os
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ===================== SAFE LOAD MODELS =====================
MODEL_LOAD_ERRORS: list[str] = []

def _p(filename: str) -> str:
    return os.path.join(BASE_DIR, filename)

# Defaults (nếu thiếu file)
DEFAULT_FEATURE_COLUMNS = [
    "monthly_revenue",
    "order_volume_30d",
    "refund_rate",
    "cashflow_volatility",
    "platform_rating",
    "years_in_business",
]

rf_model = None
logit_model = None
feature_columns = DEFAULT_FEATURE_COLUMNS

try:
    rf_model = joblib.load(_p("rf_model.pkl"))
except Exception as e:
    MODEL_LOAD_ERRORS.append(f"rf_model.pkl load failed: {e}")

try:
    feature_columns = joblib.load(_p("feature_columns.pkl"))
except Exception as e:
    MODEL_LOAD_ERRORS.append(f"feature_columns.pkl load failed: {e}")
    feature_columns = DEFAULT_FEATURE_COLUMNS

try:
    logit_model = joblib.load(_p("logit_model.pkl"))
except Exception:
    logit_model = None


# ===================== HELPERS =====================
def _to_float(x, default=0.0) -> float:
    """Robust float parsing: accepts None, '', '  ', strings, numbers."""
    try:
        if x is None:
            return float(default)
        if isinstance(x, (int, float, np.number)):
            return float(x)
        s = str(x).strip()
        if s == "":
            return float(default)
        return float(s)
    except Exception:
        return float(default)


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _get_feature_groups_from_pipeline(pipe):
    """
    Trích num_cols / cat_cols từ ColumnTransformer trong pipeline (nếu có),
    để fill default đúng kiểu (numeric=0, categorical='UNKNOWN').
    """
    if pipe is None:
        return set(), set()
    try:
        pre = getattr(pipe, "named_steps", {}).get("preprocess", None)
        if pre is None:
            return set(), set()

        transformers = getattr(pre, "transformers_", None) or getattr(pre, "transformers", None) or []
        num_cols, cat_cols = set(), set()
        for name, trans, cols in transformers:
            if name == "num":
                num_cols.update(list(cols))
            elif name == "cat":
                cat_cols.update(list(cols))
        return num_cols, cat_cols
    except Exception:
        return set(), set()


NUM_COLS, CAT_COLS = _get_feature_groups_from_pipeline(rf_model)


def build_feature_df(input_data: dict) -> pd.DataFrame:
    """
    Tạo DataFrame đúng feature space như lúc train:
    - Có đủ tất cả cột trong feature_columns
    - Thiếu numeric -> 0
    - Thiếu categorical -> 'UNKNOWN'
    """
    df = pd.DataFrame([input_data])

    for col in feature_columns:
        if col not in df.columns:
            if col in CAT_COLS:
                df[col] = "UNKNOWN"
            else:
                df[col] = 0

    df = df[list(feature_columns)].copy()

    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("UNKNOWN")

    df = df.fillna(0)
    return df


# ===================== BUSINESS BASELINE (Nghiệp vụ tín dụng) =====================
def fallback_risk_estimate(d: dict) -> float:
    """
    Ước lượng rủi ro baseline (0..1) theo nghiệp vụ.
    Mục tiêu:
      - Trả kết quả ổn định và "hợp lý nghiệp vụ" với 6 chỉ số đầu vào.
      - Dùng làm nền để hiệu chỉnh kết quả ML (RF) khi ML bị lệch do thiếu feature.
    """
    rev = d["monthly_revenue"]
    orders = d["order_volume_30d"]
    refund = d["refund_rate"]
    vol = d["cashflow_volatility"]
    rating = d["platform_rating"]
    years = d["years_in_business"]

    # Các thang này chỉ là "demo-scale": giữ nhất quán với UI
    rev_score = 1 - _clamp(rev / 120000, 0, 1)           # doanh thu càng cao rủi ro càng giảm
    order_score = 1 - _clamp(orders / 2500, 0, 1)        # số đơn càng cao rủi ro càng giảm
    refund_score = _clamp(refund / 30, 0, 1)             # hoàn đơn càng cao rủi ro càng tăng
    vol_score = _clamp(vol / 1.0, 0, 1)                  # biến động càng cao rủi ro càng tăng
    rating_score = _clamp((4.6 - rating) / 3.0, 0, 1)    # rating thấp => rủi ro tăng
    age_score = _clamp((3 - years) / 3.0, 0, 1)          # hoạt động <3 năm => rủi ro tăng

    # Trọng số nghiệp vụ (ưu tiên hoàn đơn & biến động dòng tiền)
    risk = (
        0.20 * rev_score +
        0.14 * order_score +
        0.26 * refund_score +
        0.22 * vol_score +
        0.10 * rating_score +
        0.08 * age_score
    )
    return float(_clamp(risk, 0.02, 0.98))


def choose_final_risk(
    risk_rf: float | None,
    risk_fallback: float,
    completeness: float,
) -> tuple[float, str, str | None, str]:
    """
    Chọn PD cuối theo hướng "đúng nghiệp vụ" hơn:
    - Nếu thiếu feature nhiều (completeness thấp) => RF kém tin cậy => ưu tiên baseline
    - Nếu RF lệch mạnh so với baseline => ưu tiên baseline, RF chỉ ảnh hưởng nhẹ
    """
    if risk_rf is None:
        return float(_clamp(risk_fallback, 0.02, 0.98)), "fallback", None, "medium"

    diff = abs(risk_rf - risk_fallback)

    # Độ tin cậy theo mức độ đầy đủ dữ liệu
    if completeness < 0.35:
        rf_cap = 0.20
        confidence = "low"
    elif completeness < 0.60:
        rf_cap = 0.35
        confidence = "medium"
    else:
        rf_cap = 0.55
        confidence = "high"

    # Nếu RF lệch mạnh: ưu tiên baseline (nghiệp vụ)
    if diff >= 0.35:
        rf_w = min(0.15, rf_cap)
        base_w = 1 - rf_w
        risk_final = base_w * risk_fallback + rf_w * risk_rf
        note = "ML (RF) lệch mạnh so với baseline nghiệp vụ → ưu tiên baseline để tránh chấm điểm 'oan' do thiếu feature."
        return float(_clamp(risk_final, 0.02, 0.98)), "baseline_dominant", note, confidence

    # Lệch vừa: làm mượt
    if diff >= 0.20:
        rf_w = min(0.30, rf_cap)
        base_w = 1 - rf_w
        risk_final = base_w * risk_fallback + rf_w * risk_rf
        note = "ML (RF) được hiệu chỉnh nhẹ theo baseline nghiệp vụ (do có chênh lệch vừa)."
        return float(_clamp(risk_final, 0.02, 0.98)), "calibrated", note, confidence

    # Lệch nhỏ: dùng RF nhiều hơn nhưng vẫn có baseline để ổn định
    rf_w = min(0.45, rf_cap)
    base_w = 1 - rf_w
    risk_final = base_w * risk_fallback + rf_w * risk_rf
    return float(_clamp(risk_final, 0.02, 0.98)), "rf_smoothed", None, confidence


def map_credit_tier(score: float) -> str:
    """
    Mapping theo nghiệp vụ mềm hơn để phù hợp demo SME:
    - Score ~ 55–65 thường là "cần theo dõi / thẩm định thêm"
    - Tránh trường hợp hơi dưới 40 đã bị 'D rất cao' quá gắt
    """
    if score >= 90:
        return "A+ – Rủi ro rất thấp"
    elif score >= 80:
        return "A – Rủi ro thấp"
    elif score >= 70:
        return "B+ – Khá an toàn"
    elif score >= 60:
        return "B – Trung bình"
    elif score >= 50:
        return "C+ – Cần theo dõi"
    elif score >= 35:
        return "C – Rủi ro cao"
    elif score >= 25:
        return "D – Rủi ro rất cao"
    else:
        return "E – Nguy cơ mất khả năng thanh toán"


def policy_recommendation(score: float, d: dict) -> dict:
    """
    Gợi ý nghiệp vụ tín dụng (demo):
    - Quyết định
    - Hạn mức gợi ý (cùng đơn vị với doanh thu nhập)
    - Điều kiện/thẩm định thêm
    """
    rev = float(d["monthly_revenue"])

    if score >= 70:
        decision = "NÊN PHÊ DUYỆT (rủi ro thấp)"
        limit = rev * 1.2
        conditions = [
            "Ưu tiên đối soát doanh thu sàn 3 tháng gần nhất.",
            "Theo dõi biến động dòng tiền/hoàn đơn theo tuần."
        ]
    elif score >= 50:
        decision = "CẦN THẨM ĐỊNH THÊM (rủi ro trung bình)"
        limit = rev * 0.7
        conditions = [
            "Yêu cầu sao kê/đối soát sàn 3–6 tháng.",
            "Giới hạn hạn mức ban đầu; tăng dần nếu hoàn đơn & dòng tiền ổn định."
        ]
    elif score >= 35:
        decision = "THẨM ĐỊNH CHẶT (rủi ro cao)"
        limit = rev * 0.5
        conditions = [
            "Kiểm tra kỹ hoàn đơn, khiếu nại, và dòng tiền 6 tháng.",
            "Áp điều kiện kiểm soát (giữ lại một phần doanh thu/đối soát chặt)."
        ]
    else:
        decision = "KHÔNG KHUYẾN NGHỊ (rủi ro rất cao)"
        limit = rev * 0.3
        conditions = [
            "Chỉ cân nhắc nếu có tài sản/đảm bảo hoặc đối tác bảo lãnh.",
            "Yêu cầu dữ liệu tài chính đầy đủ trước khi xem xét lại."
        ]

    return {
        "policy_decision": decision,
        "suggested_limit": round(limit, 2),
        "policy_conditions": conditions[:3],
    }


def explain_ai(d: dict, credit_score: float, risk_final: float, model_source: str, confidence: str, note: str | None):
    exps = []
    recs = []

    rev = d["monthly_revenue"]
    orders = d["order_volume_30d"]
    refund = d["refund_rate"]
    vol = d["cashflow_volatility"]
    rating = d["platform_rating"]
    years = d["years_in_business"]

    # 1) Revenue
    if rev < 30000:
        exps.append({"title": "Doanh thu TB/tháng", "impact": "Tiêu cực",
                     "detail": f"Doanh thu {rev:,.0f} thấp → biên an toàn dòng tiền mỏng, dễ hụt khi chi phí/hoàn đơn tăng."})
        recs.append("Tăng doanh thu ổn định (tăng repeat customers, tối ưu ads/CRM, nâng AOV).")
    elif rev < 80000:
        exps.append({"title": "Doanh thu TB/tháng", "impact": "Trung tính",
                     "detail": f"Doanh thu {rev:,.0f} trung bình → có dòng tiền nhưng cần kiểm soát chi phí và hoàn đơn."})
        recs.append("Giữ doanh thu đều theo tuần; theo dõi CAC/ROAS để tránh bơm ads quá mức.")
    else:
        exps.append({"title": "Doanh thu TB/tháng", "impact": "Tích cực",
                     "detail": f"Doanh thu {rev:,.0f} tốt → tăng khả năng trả nợ và hấp thụ biến động vận hành."})

    # 2) Orders
    if orders < 300:
        exps.append({"title": "Số đơn 30 ngày", "impact": "Tiêu cực",
                     "detail": f"Số đơn {orders:,.0f} thấp → rủi ro tập trung, dễ hụt dòng tiền nếu giảm bán."})
        recs.append("Tăng đơn: tối ưu listing, SEO sàn, tăng conversion & remarketing.")
    elif orders < 1200:
        exps.append({"title": "Số đơn 30 ngày", "impact": "Trung tính",
                     "detail": f"Số đơn {orders:,.0f} trung bình → cần tăng độ đều để giảm rủi ro theo mùa."})
    else:
        exps.append({"title": "Số đơn 30 ngày", "impact": "Tích cực",
                     "detail": f"Số đơn {orders:,.0f} cao → phân tán rủi ro, ổn định doanh thu."})

    # 3) Refund
    if refund > 15:
        exps.append({"title": "Tỷ lệ hoàn đơn", "impact": "Tiêu cực",
                     "detail": f"Hoàn đơn {refund:.1f}% cao → tăng chi phí vận hành, rủi ro hụt dòng tiền và giảm uy tín."})
        recs.append("Giảm hoàn: mô tả rõ, kiểm soát chất lượng, đóng gói, SLA giao hàng & CSKH.")
    elif refund > 8:
        exps.append({"title": "Tỷ lệ hoàn đơn", "impact": "Trung tính",
                     "detail": f"Hoàn đơn {refund:.1f}% cần theo dõi → có thể ảnh hưởng dòng tiền trong mùa cao điểm."})
        recs.append("Theo dõi lý do hoàn top 3 và xử lý theo nhóm nguyên nhân.")
    else:
        exps.append({"title": "Tỷ lệ hoàn đơn", "impact": "Tích cực",
                     "detail": f"Hoàn đơn {refund:.1f}% tốt → ổn định dòng tiền và giảm thất thoát doanh thu."})

    # 4) Cashflow volatility (sửa ngưỡng: 0.45 không còn bị coi là 'tích cực')
    if vol > 0.70:
        exps.append({"title": "Biến động dòng tiền", "impact": "Tiêu cực",
                     "detail": f"Biến động {vol:.2f} rất cao → khó dự báo dòng tiền, rủi ro thiếu hụt khi đến kỳ trả nợ."})
        recs.append("Lập ngân sách dòng tiền tuần/tháng; tăng dự phòng tiền mặt; hạn chế chi phí cố định.")
    elif vol >= 0.40:
        exps.append({"title": "Biến động dòng tiền", "impact": "Trung tính",
                     "detail": f"Biến động {vol:.2f} tương đối → cần quản trị tồn kho & chi phí marketing để tránh dồn vốn."})
        recs.append("Giới hạn ngân sách ads theo ROAS; tối ưu vòng quay tồn kho.")
    else:
        exps.append({"title": "Biến động dòng tiền", "impact": "Tích cực",
                     "detail": f"Biến động {vol:.2f} thấp → dòng tiền ổn định, phù hợp triển khai hạn mức."})

    # 5) Platform rating
    if rating < 3.8:
        exps.append({"title": "Rating sàn TMĐT", "impact": "Tiêu cực",
                     "detail": f"Rating {rating:.1f} thấp → tăng rủi ro giảm đơn & tăng hoàn/khiếu nại."})
        recs.append("Nâng rating: cải thiện đóng gói, giao đúng mô tả, phản hồi CSKH nhanh.")
    elif rating < 4.3:
        exps.append({"title": "Rating sàn TMĐT", "impact": "Trung tính",
                     "detail": f"Rating {rating:.1f} ổn → nên cải thiện thêm để tăng trust và giảm chi phí chuyển đổi."})
    else:
        exps.append({"title": "Rating sàn TMĐT", "impact": "Tích cực",
                     "detail": f"Rating {rating:.1f} tốt → tăng niềm tin và giảm rủi ro vận hành."})

    # 6) Years in business
    if years < 1:
        exps.append({"title": "Thời gian hoạt động", "impact": "Tiêu cực",
                     "detail": f"Hoạt động {years:.1f} năm → lịch sử ít, rủi ro mô hình kinh doanh chưa ổn định."})
        recs.append("Tăng minh bạch: sao kê doanh thu, đối soát sàn, lịch sử giao dịch để nâng độ tin cậy.")
    elif years < 3:
        exps.append({"title": "Thời gian hoạt động", "impact": "Trung tính",
                     "detail": f"Hoạt động {years:.1f} năm → có lịch sử nhưng cần chứng minh ổn định qua nhiều mùa bán."})
    else:
        exps.append({"title": "Thời gian hoạt động", "impact": "Tích cực",
                     "detail": f"Hoạt động {years:.1f} năm → độ bền tốt, giảm rủi ro 'mở ra đóng vào'."})

    # Summary theo NGHIỆP VỤ: bám credit_score (không dùng mỗi 'đếm tiêu cực' nữa)
    if credit_score >= 70:
        summary = "Rủi ro thấp: có thể phê duyệt hạn mức phù hợp, ưu tiên kiểm soát biến động dòng tiền và duy trì tỷ lệ hoàn thấp."
    elif credit_score >= 50:
        summary = "Rủi ro trung bình: nên thẩm định thêm, cấp hạn mức ban đầu vừa phải và theo dõi hoàn đơn/dòng tiền 1–2 chu kỳ."
    elif credit_score >= 35:
        summary = "Rủi ro cao: thẩm định chặt, ưu tiên hạn mức thấp và yêu cầu đối soát/sao kê trước khi cấp."
    else:
        summary = "Rủi ro rất cao: không khuyến nghị cấp tín dụng nếu thiếu đảm bảo hoặc thiếu dữ liệu tài chính."

    # Thêm ghi chú hiệu chỉnh nếu có
    if note:
        summary += f"\n\n(Ghi chú mô hình: {note})"
    summary += f"\n(Độ tin cậy: {confidence})"

    recs = list(dict.fromkeys(recs))
    return exps[:7], recs[:5], summary


# ===================== ROUTES =====================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/score", methods=["POST"], strict_slashes=False)
def score_api():
    try:
        payload = request.get_json(silent=True) or {}

        # SANITIZE INPUT
        data = {
            "monthly_revenue": _clamp(_to_float(payload.get("monthly_revenue"), 0), 0, 10_000_000_000),
            "order_volume_30d": _clamp(_to_float(payload.get("order_volume_30d"), 0), 0, 10_000_000),
            "refund_rate": _clamp(_to_float(payload.get("refund_rate"), 0), 0, 100),
            "cashflow_volatility": _clamp(_to_float(payload.get("cashflow_volatility"), 0.3), 0, 1),
            "platform_rating": _clamp(_to_float(payload.get("platform_rating"), 4.0), 1, 5),
            "years_in_business": _clamp(_to_float(payload.get("years_in_business"), 0), 0, 100),
        }

        df = build_feature_df(data)

        # Data completeness vs training feature space
        present = sum(1 for c in feature_columns if c in data)
        completeness = present / max(1, len(feature_columns))

        # Business baseline
        risk_fallback = float(fallback_risk_estimate(data))

        # RF prediction (if available)
        risk_rf = None
        if rf_model is not None:
            try:
                risk_rf = float(rf_model.predict_proba(df)[0][1])
                if not np.isfinite(risk_rf):
                    risk_rf = None
            except Exception:
                risk_rf = None

        # Logit prediction (for display only; NOT used in final risk)
        risk_logit = None
        if logit_model is not None:
            try:
                risk_logit = float(logit_model.predict_proba(df)[0][1])
                if not np.isfinite(risk_logit):
                    risk_logit = None
            except Exception:
                risk_logit = None

        # Choose final risk (nghiệp vụ)
        risk_final, model_source, calibration_note, confidence = choose_final_risk(
            risk_rf=risk_rf,
            risk_fallback=risk_fallback,
            completeness=completeness,
        )

        credit_score = round((1 - risk_final) * 100, 1)
        tier = map_credit_tier(credit_score)

        # Explanations + Summary (nghiệp vụ)
        explanations, recommendations, summary = explain_ai(
            data,
            credit_score=credit_score,
            risk_final=risk_final,
            model_source=model_source,
            confidence=confidence,
            note=calibration_note,
        )

        # Policy recommendation
        policy = policy_recommendation(credit_score, data)

        resp = {
            "success": True,
            "credit_score": credit_score,
            "tier": tier,

            "risk_prob_final": round(float(risk_final), 4),
            "risk_prob_fallback": round(float(risk_fallback), 4),
            "risk_prob_rf": (round(float(risk_rf), 4) if risk_rf is not None else None),
            "risk_prob_logit": (round(float(risk_logit), 4) if risk_logit is not None else None),

            "model_source": model_source,
            "model_confidence": confidence,
            "calibration_note": calibration_note,

            "summary": summary,
            "explanations": explanations,
            "recommendations": recommendations,

            "policy_decision": policy.get("policy_decision"),
            "suggested_limit": policy.get("suggested_limit"),
            "policy_conditions": policy.get("policy_conditions"),

            "sanitized_input": data,
            "feature_completeness": round(float(completeness), 3),
        }

        if MODEL_LOAD_ERRORS:
            resp["model_warnings"] = MODEL_LOAD_ERRORS[:3]

        return jsonify(resp)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
