"""
Lab #3: Baseline Chatbot vs ReAct Agent
VinUni AI Training Program — Day 03

Task 1: class ChatbotBaseline — trả lời 1 lượt, KHÔNG dùng tool.
Task 2: class ReActAgent      — vòng lặp Thought → Action → Observation + safeguards.

Cấu hình: điền GEMINI_API_KEY vào file .env ở thư mục gốc dự án.
Để trống -> agent chạy chế độ offline (rule-based), kết quả hoàn toàn deterministic.
"""

import json
import os
import re
import warnings
from typing import Any, Dict, List, Tuple

from tools import TOOL_DEFINITIONS, TOOL_MAP

# Nạp GEMINI_API_KEY từ .env ở thư mục gốc dự án (chạy được từ mọi cwd).
try:
    from dotenv import load_dotenv

    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    load_dotenv()  # dự phòng: tìm .env theo cwd
except ImportError:
    pass

GEMINI_MODEL = "gemini-1.5-flash"

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

# --- Bộ nhận diện ý định (intent) cho phần suy luận offline ---
AIRPORT_PATTERN = re.compile(r"\b(?:HAN|SGN|DAD)\b")
CITY_ALIASES: Tuple[Tuple[str, str], ...] = (
    ("HÀ NỘI", "HAN"),
    ("HỒ CHÍ MINH", "SGN"),
    ("SÀI GÒN", "SGN"),
    ("ĐÀ NẴNG", "DAD"),
)
FLIGHT_KEYWORDS = ("chuyến bay", "vé máy bay", "vé bay", "giá vé", "flight")
WEATHER_KEYWORDS = ("thời tiết", "mặc gì", "nhiệt độ", "mưa", "nắng", "weather")

FAQ_ANSWER = (
    "Về chính sách đổi/trả vé của Vinpearl: vé có thể đổi ngày bay trước giờ khởi hành "
    "24 giờ với phí đổi theo điều kiện vé; vé khuyến mại thường không được hoàn. "
    "Vui lòng liên hệ hotline Vinpearl để được xác nhận chi tiết theo mã đặt chỗ của bạn."
)

BASELINE_FALLBACK = (
    "Xin lỗi, tôi không có kết nối tới hệ thống tra cứu chuyến bay hay dữ liệu thời tiết "
    "thời gian thực, nên không thể xác nhận giá vé cụ thể. Theo thông tin tham khảo, vé "
    "HAN–SGN thường dao động 1–3 triệu VND (có thể không chính xác)."
)


def call_gemini(api_key: str, prompt: str) -> str:
    """Gọi Gemini 1 lượt. Trả về chuỗi rỗng nếu lỗi/thiếu SDK (để agent chạy offline)."""
    try:
        with warnings.catch_warnings():
            # SDK google-generativeai đã deprecated -> tắt FutureWarning cho gọn log
            warnings.simplefilter("ignore", FutureWarning)
            import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        return (response.text or "").strip()
    except Exception:
        return ""


class ChatbotBaseline:
    """Baseline LLM Chatbot without ReAct Loop or Tools"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def query(self, user_input: str) -> Dict[str, Any]:
        answer = ""
        if self.api_key:
            answer = call_gemini(
                self.api_key,
                "Bạn là chatbot tư vấn du lịch. Hãy trả lời ngắn gọn, "
                f"KHÔNG dùng tool hay internet: {user_input}",
            )
        if not answer:
            answer = BASELINE_FALLBACK

        return {
            "status": "success",
            "answer": answer,
            "tool_calls": [],  # ← khác biệt cốt lõi so với ReAct Agent
            "iterations": 1,
        }


class ReActAgent:
    """Production-grade ReAct Agent with Tool Registry and Safeguards"""

    def __init__(self, max_iterations: int = 5, api_key: str = None):
        self.max_iterations = max_iterations
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.trace: List[Dict[str, Any]] = []
        self.results: Dict[str, Any] = {}
        self.system_prompt = SYSTEM_PROMPT.format(
            tools=json.dumps(TOOL_DEFINITIONS, ensure_ascii=False, indent=2)
        )

    # ------------------------------------------------------------------
    # Parsers — bóc tham số tool từ câu nói tự nhiên
    # ------------------------------------------------------------------
    def parse_city_code(self, text: str) -> str:
        """'Đà Nẵng DAD' -> 'DAD' | 'thời tiết Hà Nội' -> 'HAN'. Mặc định 'SGN'."""
        text_upper = text.upper()
        match = AIRPORT_PATTERN.search(text_upper)
        if match:
            return match.group(0)
        for alias, code in CITY_ALIASES:
            if alias in text_upper:
                return code
        return "SGN"

    def parse_route(self, text: str) -> Tuple[str, str]:
        """'từ HAN đi SGN' -> ('HAN', 'SGN'). Thiếu mã thì trả về ('', '')."""
        codes = AIRPORT_PATTERN.findall(text.upper())
        if len(codes) >= 2:
            return codes[0], codes[1]
        return "", ""

    def parse_max_price(self, text: str, default: int = 5_000_000) -> int:
        """'dưới 2 triệu' -> 2000000 | '1.5 triệu' -> 1500000 | '500k' -> 500000."""
        match = re.search(
            r"(\d+(?:[.,]\d+)?)\s*(triệu|tr\b|k\b|nghìn|ngàn)", text.lower()
        )
        if not match:
            return default
        value = float(match.group(1).replace(",", "."))
        unit = match.group(2).strip()
        return int(value * 1_000_000) if unit.startswith("tr") else int(value * 1_000)

    def has_city(self, text: str) -> bool:
        text_upper = text.upper()
        return bool(AIRPORT_PATTERN.search(text_upper)) or any(
            alias in text_upper for alias, _ in CITY_ALIASES
        )

    # ------------------------------------------------------------------
    # Planner — mô phỏng bước Thought: cần gọi những tool nào?
    # ------------------------------------------------------------------
    def plan_tools(self, user_input: str) -> List[Dict[str, Any]]:
        text = user_input.lower()
        origin, destination = self.parse_route(user_input)
        plan: List[Dict[str, Any]] = []

        if any(kw in text for kw in FLIGHT_KEYWORDS) and origin and destination:
            plan.append(
                {
                    "name": "get_flight_info",
                    "args": {
                        "origin": origin,
                        "destination": destination,
                        "max_price": self.parse_max_price(user_input),
                    },
                }
            )

        if any(kw in text for kw in WEATHER_KEYWORDS) and self.has_city(user_input):
            city = destination or self.parse_city_code(user_input)
            plan.append({"name": "get_weather_forecast", "args": {"city_code": city}})

        return plan

    # ------------------------------------------------------------------
    # Executor — Action (JSON) -> Observation. Chống Trap 1 & Trap 2.
    # ------------------------------------------------------------------
    def execute_tool(self, action: Any) -> Any:
        try:
            payload = action if isinstance(action, dict) else json.loads(action)
        except (json.JSONDecodeError, TypeError):
            return {"error": "Invalid JSON format"}  # Trap 2: format drift

        name = str(payload.get("name", "")).strip().lower()  # Trap 1: tên tool bẩn
        args = payload.get("args") or {}

        if name not in TOOL_MAP:
            return {"error": f"Unknown tool: {name}"}
        try:
            return TOOL_MAP[name](**args)
        except TypeError as exc:
            return {"error": f"Invalid arguments: {exc}"}

    # ------------------------------------------------------------------
    # Synthesizer — ghép các Observation thành Final Answer
    # ------------------------------------------------------------------
    def compose_answer(self) -> str:
        parts: List[str] = []

        flights = self.results.get("get_flight_info")
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

        weather = self.results.get("get_weather_forecast")
        if isinstance(weather, dict) and "error" not in weather:
            parts.append(
                f"Thời tiết {weather['city']}: {weather['temperature_c']}°C, "
                f"{weather['condition']}, độ ẩm {weather['humidity_pct']}%. "
                f"Gợi ý trang phục: {weather['recommendation']}"
            )
        elif isinstance(weather, dict):
            parts.append(f"Chưa lấy được dữ liệu thời tiết: {weather['error']}")

        return "\n\n".join(parts) if parts else FAQ_ANSWER

    def build_thought(self, user_input: str, next_tool: str) -> str:
        """Sinh câu Thought. Có API key thì nhờ Gemini diễn giải, không thì dùng mẫu."""
        template = (
            f"Tôi cần gọi `{next_tool}` để lấy dữ liệu còn thiếu."
            if next_tool
            else "Đã có đủ dữ liệu, tổng hợp câu trả lời cuối cùng."
        )
        if not self.api_key:
            return template
        thought = call_gemini(
            self.api_key,
            f"{self.system_prompt}\n\nCâu hỏi: {user_input}\n"
            f"Tool sắp gọi: {next_tool or '(không, đã đủ dữ liệu)'}\n"
            "Viết DUY NHẤT một câu Thought tiếng Việt, không thêm gì khác.",
        )
        return thought or template

    # ------------------------------------------------------------------
    # Một bước ReAct: trả về (answer, is_done)
    # ------------------------------------------------------------------
    def plan_and_execute_step(self, user_input: str, iteration: int) -> Tuple[str, bool]:
        plan = self.plan_tools(user_input)
        pending = [a for a in plan if a["name"] not in self.results]

        # Không còn tool nào cần gọi -> bước tổng hợp Final Answer
        if not pending:
            answer = self.compose_answer()
            self.trace.append(
                {
                    "iteration": iteration,
                    "thought": self.build_thought(user_input, ""),
                    "final_answer": answer,
                }
            )
            return answer, True

        # Thought -> Action -> Observation
        action = pending[0]
        entry = {
            "iteration": iteration,
            "thought": self.build_thought(user_input, action["name"]),
            "action": action,
            "observation": self.execute_tool(action),
        }
        # Trap 3: ghi kết quả kể cả khi lỗi -> pending luôn co lại, không lặp vô tận
        self.results[action["name"]] = entry["observation"]
        self.trace.append(entry)

        # Chỉ cần 1 tool -> đã đủ dữ liệu, trả lời luôn trong bước này
        if len(plan) == 1:
            answer = self.compose_answer()
            entry["final_answer"] = answer
            return answer, True

        return "", False

    # ------------------------------------------------------------------
    # ReAct Loop + Safeguard
    # ------------------------------------------------------------------
    def run(self, user_input: str) -> Dict[str, Any]:
        self.trace = []
        self.results = {}
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            answer, is_done = self.plan_and_execute_step(user_input, iteration)
            if is_done:
                return {
                    "status": "completed",
                    "answer": answer,
                    "iterations": iteration,
                    "trace": self.trace,
                }

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

    mode = "GEMINI" if os.getenv("GEMINI_API_KEY") else "OFFLINE (rule-based)"
    print(f"### CHẾ ĐỘ SUY LUẬN: {mode} ###\n")

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
