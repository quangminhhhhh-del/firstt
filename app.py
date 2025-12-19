from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS

import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)


#  LOAD MODELS 
rf_model = joblib.load("rf_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# Optional: load logistic model if exists (không bắt buộc)
try:
    logit_model = joblib.load("logit_model.pkl")
except Exception:
    logit_model = None


#  HELPERS 
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
    try:
        pre = pipe.named_steps.get("preprocess", None)
        if pre is None:
            return set(), set()

        # transformers có thể là transformers_ hoặc transformers
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

    df = df[feature_columns].copy()

    # Fix dtype cho categorical để pipeline OneHot chạy ổn định
    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("UNKNOWN")

    # Fill NaN numeric
    df = df.fillna(0)
    return df


def fallback_risk_estimate(d: dict) -> float:
    """
    Ước lượng rủi ro fallback (0..1) khi model lỗi.
    Mục tiêu: LUÔN trả kết quả, không cần chính xác tuyệt đối.
    """
    rev = d["monthly_revenue"]
    orders = d["order_volume_30d"]
    refund = d["refund_rate"]
    vol = d["cashflow_volatility"]
    rating = d["platform_rating"]
    years = d["years_in_business"]

    # Chuẩn hoá về thang tương đối
    rev_score = 1 - _clamp(rev / 120000, 0, 1)          # doanh thu cao -> giảm rủi ro
    order_score = 1 - _clamp(orders / 2500, 0, 1)       # đơn nhiều -> giảm rủi ro
    refund_score = _clamp(refund / 30, 0, 1)            # hoàn đơn cao -> tăng rủi ro
    vol_score = _clamp(vol / 1.0, 0, 1)                 # biến động cao -> tăng rủi ro
    rating_score = _clamp((4.6 - rating) / 3.0, 0, 1)   # rating thấp -> tăng rủi ro
    age_score = _clamp((3 - years) / 3.0, 0, 1)         # hoạt động ngắn -> tăng rủi ro

    # trọng số 
    risk = (
        0.22 * rev_score +
        0.14 * order_score +
        0.24 * refund_score +
        0.20 * vol_score +
        0.12 * rating_score +
        0.08 * age_score
    )

    return float(_clamp(risk, 0.02, 0.98))


def map_credit_tier(score: float) -> str:
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
    elif score >= 40:
        return "C – Rủi ro cao"
    elif score >= 30:
        return "D – Rủi ro rất cao"
    else:
        return "E – Nguy cơ mất khả năng thanh toán"


def explain_ai(d: dict) -> tuple[list[dict], list[str], str]:
    """
    Trả về:
      - explanations: list[{title, impact, detail}]
      - recommendations: list[str]
      - summary: str
    """

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
        exps.append({
            "title": "Doanh thu TB/tháng",
            "impact": "Tiêu cực",
            "detail": f"Doanh thu {rev:,.0f} khá thấp → biên an toàn dòng tiền mỏng, dễ bị sốc khi chi phí tăng/đơn hoàn nhiều."
        })
        recs.append("Tăng doanh thu ổn định (tăng repeat customers, tối ưu ads/CRM, nâng AOV).")
    elif rev < 80000:
        exps.append({
            "title": "Doanh thu TB/tháng",
            "impact": "Trung tính",
            "detail": f"Doanh thu {rev:,.0f} ở mức trung bình → có dòng tiền nhưng vẫn cần kiểm soát chi phí/hoàn đơn để giữ khả năng trả nợ."
        })
        recs.append("Giữ doanh thu đều theo tuần; theo dõi CAC/ROAS để tránh bơm ads quá mức.")
    else:
        exps.append({
            "title": "Doanh thu TB/tháng",
            "impact": "Tích cực",
            "detail": f"Doanh thu {rev:,.0f} tốt → tăng khả năng tạo dòng tiền trả nợ và hấp thụ biến động vận hành."
        })

    # 2) Orders
    if orders < 300:
        exps.append({
            "title": "Số đơn 30 ngày",
            "impact": "Tiêu cực",
            "detail": f"Số đơn {orders:,.0f} thấp → phụ thuộc ít đơn, rủi ro tập trung (chỉ cần vài tuần kém là dòng tiền hụt)."
        })
        recs.append("Tăng lượng đơn bằng tối ưu listing, SEO sàn, tăng conversion & remarketing.")
    elif orders < 1200:
        exps.append({
            "title": "Số đơn 30 ngày",
            "impact": "Trung tính",
            "detail": f"Số đơn {orders:,.0f} trung bình → có hoạt động bán, nhưng cần tăng độ đều để giảm rủi ro theo mùa."
        })
    else:
        exps.append({
            "title": "Số đơn 30 ngày",
            "impact": "Tích cực",
            "detail": f"Số đơn {orders:,.0f} cao → doanh thu phân tán nhiều đơn, giảm rủi ro biến động đột ngột."
        })

    # 3) Refund
    if refund > 15:
        exps.append({
            "title": "Tỷ lệ hoàn đơn",
            "impact": "Tiêu cực",
            "detail": f"Hoàn đơn {refund:.1f}% cao → tăng chi phí vận hành, tăng rủi ro hụt dòng tiền và giảm uy tín trên sàn."
        })
        recs.append("Giảm hoàn: mô tả sản phẩm rõ, kiểm soát chất lượng, đóng gói, SLA giao hàng & CSKH.")
    elif refund > 8:
        exps.append({
            "title": "Tỷ lệ hoàn đơn",
            "impact": "Trung tính",
            "detail": f"Hoàn đơn {refund:.1f}% cần theo dõi → vẫn có thể ảnh hưởng dòng tiền nếu tăng thêm trong mùa cao điểm."
        })
        recs.append("Theo dõi lý do hoàn top 3, xử lý theo nhóm nguyên nhân (size, lỗi, giao chậm).")
    else:
        exps.append({
            "title": "Tỷ lệ hoàn đơn",
            "impact": "Tích cực",
            "detail": f"Hoàn đơn {refund:.1f}% ở mức tốt → giảm thất thoát doanh thu và ổn định dòng tiền."
        })

    # 4) Cashflow volatility
    if vol > 0.7:
        exps.append({
            "title": "Biến động dòng tiền",
            "impact": "Tiêu cực",
            "detail": f"Biến động {vol:.2f} rất cao → dòng tiền khó dự báo, rủi ro thiếu hụt khi đến kỳ trả nợ."
        })
        recs.append("Thiết lập ngân sách dòng tiền tuần/tháng; tăng dự phòng tiền mặt; hạn chế chi phí cố định.")
    elif vol > 0.45:
        exps.append({
            "title": "Biến động dòng tiền",
            "impact": "Trung tính",
            "detail": f"Biến động {vol:.2f} tương đối → có dao động theo mùa/ads; cần quản trị tồn kho & chi phí marketing."
        })
        recs.append("Giới hạn ngân sách ads theo ROAS; tối ưu vòng quay tồn kho để tránh dồn vốn.")
    else:
        exps.append({
            "title": "Biến động dòng tiền",
            "impact": "Tích cực",
            "detail": f"Biến động {vol:.2f} thấp → dòng tiền ổn định, phù hợp triển khai hạn mức tín dụng."
        })

    # 5) Platform rating
    if rating < 3.8:
        exps.append({
            "title": "Rating sàn TMĐT",
            "impact": "Tiêu cực",
            "detail": f"Rating {rating:.1f} thấp → phản ánh trải nghiệm khách hàng chưa tốt, có thể kéo giảm đơn & tăng hoàn/khách khiếu nại."
        })
        recs.append("Nâng rating: cải thiện đóng gói, giao đúng mô tả, phản hồi CSKH nhanh, xử lý khiếu nại.")
    elif rating < 4.3:
        exps.append({
            "title": "Rating sàn TMĐT",
            "impact": "Trung tính",
            "detail": f"Rating {rating:.1f} ở mức ổn → vẫn nên cải thiện để tăng trust và giảm chi phí chuyển đổi."
        })
    else:
        exps.append({
            "title": "Rating sàn TMĐT",
            "impact": "Tích cực",
            "detail": f"Rating {rating:.1f} tốt → tăng niềm tin, hỗ trợ tăng conversion và giảm rủi ro kinh doanh."
        })

    # 6) Years in business
    if years < 1:
        exps.append({
            "title": "Thời gian hoạt động",
            "impact": "Tiêu cực",
            "detail": f"Hoạt động {years:.1f} năm → dữ liệu lịch sử ít, rủi ro mô hình kinh doanh chưa ổn định."
        })
        recs.append("Tăng minh bạch: sao kê doanh thu, hợp đồng NCC, lịch sử giao dịch để nâng độ tin cậy.")
    elif years < 3:
        exps.append({
            "title": "Thời gian hoạt động",
            "impact": "Trung tính",
            "detail": f"Hoạt động {years:.1f} năm → có lịch sử nhưng vẫn cần chứng minh tính ổn định qua nhiều mùa bán."
        })
    else:
        exps.append({
            "title": "Thời gian hoạt động",
            "impact": "Tích cực",
            "detail": f"Hoạt động {years:.1f} năm → có độ bền, giảm rủi ro “mở ra đóng vào”."
        })

    
    neg = sum(1 for e in exps if e["impact"] == "Tiêu cực")
    if neg >= 3:
        summary = "Doanh nghiệp có nhiều tín hiệu rủi ro; nên cấp hạn mức thấp hoặc yêu cầu thêm chứng từ/điều kiện kiểm soát."
    elif neg == 2:
        summary = "Doanh nghiệp ở mức trung bình; có thể cấp hạn mức vừa phải và theo dõi các chỉ số hoàn đơn/dòng tiền."
    else:
        summary = "Doanh nghiệp có tín hiệu tốt; có thể xem xét cấp hạn mức phù hợp và ưu tiên duy trì ổn định dòng tiền."

    # Loại bỏ trùng khuyến nghị
    recs = list(dict.fromkeys(recs))

    return exps[:6], recs[:4], summary


# ROUTES 
from flask import render_template, request, jsonify

@app.route("/")
def home():
    return render_template("index.html")


# route score
@app.route("/score", methods=["POST"])
def score_api():

    payload = request.get_json(silent=True) or {}  # nhận dữ liệu từ frontend

    # SANITIZE INPUT: luôn hợp lệ 
    data = {
        "monthly_revenue": _clamp(_to_float(payload.get("monthly_revenue"), 0), 0, 10_000_000_000),
        "order_volume_30d": _clamp(_to_float(payload.get("order_volume_30d"), 0), 0, 10_000_000),
        "refund_rate": _clamp(_to_float(payload.get("refund_rate"), 0), 0, 100),
        "cashflow_volatility": _clamp(_to_float(payload.get("cashflow_volatility"), 0.3), 0, 1),
        "platform_rating": _clamp(_to_float(payload.get("platform_rating"), 4.0), 1, 5),
        "years_in_business": _clamp(_to_float(payload.get("years_in_business"), 0), 0, 100),
    }

    df = build_feature_df(data)

    #  PREDICT (primary) + fallback (guarantee) 
    model_source = "rf_model"
    try:
        risk_rf = float(rf_model.predict_proba(df)[0][1])
    except Exception:
        risk_rf = fallback_risk_estimate(data)
        model_source = "fallback"

    risk_logit = None
    if logit_model is not None:
        try:
            risk_logit = float(logit_model.predict_proba(df)[0][1])
        except Exception:
            risk_logit = None

    credit_score = round((1 - risk_rf) * 100, 1)
    tier = map_credit_tier(credit_score)

    explanations, recommendations, summary = explain_ai(data)

    return jsonify({
        "success": True,
        "credit_score": credit_score,
        "tier": tier,
        "risk_prob_rf": round(risk_rf, 4),
        "risk_prob_logit": (round(risk_logit, 4) if risk_logit is not None else None),
        "model_source": model_source,
        "summary": summary,
        "explanations": explanations,
        "recommendations": recommendations,
        "sanitized_input": data
    })


if __name__ == "__main__":
   app.run(host="0.0.0.0", port=5000, debug=True)
