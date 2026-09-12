"""
Lab #3: Baseline Chatbot vs ReAct Agent
"""

import json
import re

from tools import TOOL_DEFINITIONS, TOOL_MAP

SYSTEM_PROMPT = """Bạn là một ReAct Agent thông minh hỗ trợ khách hàng Vingroup.
Bạn chỉ sử dụng các công cụ sau:
{tools}

Quy trình trả lời bắt buộc:
Thought: <Suy nghĩ bước tiếp theo>
Action: {{"name": "<tên tool>", "args": {{<tham số>}}}}
Observation: <Kết quả từ tool>
... (Lặp lại cho tới khi có đủ dữ liệu)
Final Answer: <Câu trả lời hoàn chỉnh cho khách hàng>
"""

# --- Từ khóa để "LLM giả lập" nhận diện ý định người dùng ---
AIRPORT_PATTERN = re.compile(r"\b(?:HAN|SGN|DAD)\b")
FLIGHT_KEYWORDS = ("chuyến bay", "vé máy bay", "vé bay", "giá vé", "flight")
WEATHER_KEYWORDS = ("thời tiết", "mặc gì", "nhiệt độ", "mưa", "nắng", "weather")

FAQ_ANSWER = (
    "Về chính sách đổi/trả vé của Vinpearl: vé có thể đổi ngày bay trước giờ khởi hành "
    "24 giờ với phí đổi theo điều kiện vé; vé khuyến mại thường không được hoàn. "
    "Vui lòng liên hệ hotline Vinpearl để được xác nhận chi tiết theo mã đặt chỗ của bạn."
)

class ChatbotBaseline:
    """Baseline LLM Chatbot — trả lời 1 lượt, KHÔNG gọi tool."""

    def query(self, user_input: str) -> dict:
        answer = (
            "Xin lỗi, tôi không có kết nối tới hệ thống tra cứu chuyến bay hay "
            "dữ liệu thời tiết thời gian thực, nên không thể xác nhận giá vé cụ thể. "
            "Theo thông tin tham khảo, vé HAN–SGN thường dao động 1–3 triệu VND "
            "(có thể không chính xác)."
        )
        return {
            "status": "success",
            "answer": answer,
            "tool_calls": [],   # ← điểm khác biệt cốt lõi so với ReAct Agent
            "iterations": 1,
        }
def extract_airport_codes(text: str) -> list:
    """Trích mã sân bay theo thứ tự xuất hiện: 'HAN đi SGN' -> ['HAN', 'SGN']."""
    return AIRPORT_PATTERN.findall(text.upper())


def extract_max_price(text: str, default: int = 5_000_000) -> int:
    """'dưới 2 triệu' -> 2000000 | '1.5 triệu' -> 1500000 | '500k' -> 500000."""
    match = re.search(r"(\d+(?:[.,]\d+)?)\s*(triệu|tr\b|k\b|nghìn|ngàn)", text.lower())
    if not match:
        return default
    value = float(match.group(1).replace(",", "."))
    unit = match.group(2).strip()
    return int(value * 1_000_000) if unit.startswith("tr") else int(value * 1_000)

class ReActAgent:
    """ReAct Agent với vòng lặp Thought → Action → Observation."""

    def __init__(self, max_iterations: int = 5):
        self.max_iterations = max_iterations
        self.trace = []
        self.system_prompt = SYSTEM_PROMPT.format(
            tools=json.dumps(TOOL_DEFINITIONS, ensure_ascii=False, indent=2)
        )

    # ---------- 1. Planner: mô phỏng bước Thought của LLM ----------
    def _plan(self, user_input: str) -> list:
        """Phân tích ý định -> danh sách Action cần thực hiện."""
        text = user_input.lower()
        codes = extract_airport_codes(user_input)
        plan = []

        if any(kw in text for kw in FLIGHT_KEYWORDS) and len(codes) >= 2:
            plan.append({
                "name": "get_flight_info",
                "args": {
                    "origin": codes[0],
                    "destination": codes[1],
                    "max_price": extract_max_price(text),
                },
            })

        if any(kw in text for kw in WEATHER_KEYWORDS) and codes:
            plan.append({
                "name": "get_weather_forecast",
                "args": {"city_code": codes[1] if len(codes) >= 2 else codes[0]},
            })

        return plan

    # ---------- 2. Executor: gọi tool thật + chống 2 trap đầu ----------
    def _execute(self, action) -> object:
        try:
            payload = action if isinstance(action, dict) else json.loads(action)
        except (json.JSONDecodeError, TypeError):
            return {"error": "Invalid JSON format"}      # Trap 2

        name = str(payload.get("name", "")).strip().lower()   # Trap 1
        args = payload.get("args") or {}

        if name not in TOOL_MAP:
            return {"error": f"Unknown tool: {name}"}
        try:
            return TOOL_MAP[name](**args)
        except TypeError as exc:
            return {"error": f"Invalid arguments: {exc}"}

    # ---------- 3. Synthesizer: ghép Observation -> Final Answer ----------
    def _compose(self, results: dict) -> str:
        parts = []

        flights = results.get("get_flight_info")
        if flights is not None:
            if isinstance(flights, list) and flights:
                lines = [
                    f"- {f['flight_number']} ({f['airline']}) khởi hành {f['departure_time']}, "
                    f"giá {f['price_vnd']:,} VND"
                    for f in sorted(flights, key=lambda x: x["price_vnd"])
                ]
                parts.append("Các chuyến bay phù hợp:\n" + "\n".join(lines))
            else:
                parts.append(
                    "Rất tiếc, hiện không có chuyến bay nào khớp với yêu cầu của bạn. "
                    "Bạn có thể thử nới ngân sách hoặc đổi ngày bay."
                )

        weather = results.get("get_weather_forecast")
        if isinstance(weather, dict) and "error" not in weather:
            parts.append(
                f"Thời tiết {weather['city']}: {weather['temperature_c']}°C, "
                f"{weather['condition']}, độ ẩm {weather['humidity_pct']}%. "
                f"Gợi ý trang phục: {weather['recommendation']}"
            )
        elif isinstance(weather, dict):
            parts.append(f"Chưa lấy được dữ liệu thời tiết: {weather['error']}")

        return "\n\n".join(parts) if parts else FAQ_ANSWER

    # ---------- 4. ReAct Loop ----------
    def run(self, user_input: str) -> dict:
        self.trace = []                       # TODO 1
        plan = self._plan(user_input)
        results = {}
        iteration = 0

        while iteration < self.max_iterations:          # TODO 2
            iteration += 1
            pending = [a for a in plan if a["name"] not in results]

            # --- Không còn tool cần gọi -> Final Answer ---
            if not pending:                             # TODO 3
                answer = self._compose(results)
                self.trace.append({
                    "iteration": iteration,
                    "thought": "Đã có đủ dữ liệu, tổng hợp câu trả lời cuối cùng.",
                    "final_answer": answer,
                })
                return {
                    "status": "completed",
                    "answer": answer,
                    "iterations": iteration,
                    "trace": self.trace,
                }

            # --- Thought -> Action -> Observation ---
            action = pending[0]
            thought = f"Tôi cần gọi `{action['name']}` để lấy dữ liệu còn thiếu."
            observation = self._execute(action)          # TODO 4
            results[action["name"]] = observation

            entry = {                                    # TODO 5
                "iteration": iteration,
                "thought": thought,
                "action": action,
                "observation": observation,
            }
            self.trace.append(entry)

            # Chỉ cần 1 tool -> đã đủ dữ liệu, trả lời ngay trong bước này
            if len(plan) == 1:
                answer = self._compose(results)
                entry["final_answer"] = answer
                return {
                    "status": "completed",
                    "answer": answer,
                    "iterations": iteration,
                    "trace": self.trace,
                }

        # Milestone 4: Safeguard chống lặp vô tận
        return {
            "status": "max_iterations_reached",
            "answer": "Không thể hoàn thành yêu cầu trong số bước tối đa cho phép.",
            "iterations": iteration,
            "trace": self.trace,
        }


def main():
    queries = [
        "Tìm cho tôi chuyến bay từ HAN đi SGN dưới 2 triệu, rồi cho biết thời tiết SGN nên mặc gì?",
        "Có chuyến bay nào từ HAN đi DAD giá dưới 1.5 triệu không?",
        "Thời tiết ở Đà Nẵng DAD hiện tại thế nào?",
        "Chính sách đổi trả vé máy bay Vinpearl như thế nào?",
        "Tìm cho tôi chuyến bay từ SGN đi HAN dưới 500k.",
    ]

    print("=== CHATBOT BASELINE (không tool) ===")
    print(ChatbotBaseline().query(queries[0])["answer"])

    for q in queries:
        print("\n" + "=" * 70)
        print(f"USER: {q}")
        result = ReActAgent(max_iterations=5).run(q)
        print(f"STATUS: {result['status']} | ITERATIONS: {result['iterations']}")
        print(f"ANSWER:\n{result['answer']}")
        print("TRACE:", json.dumps(result["trace"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()