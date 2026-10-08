from pathlib import Path

EXAMPLES = [
"""# Python function
def add(a: int, b: int) -> int:
    return a + b

def multiply(a: int, b: int) -> int:
    return a * b
""",
"""# Python validation
def is_valid_email(value: str) -> bool:
    return "@" in value and "." in value.split("@")[-1]

def is_positive(value: int) -> bool:
    return value > 0
""",
"""# Python data processing
def total(values: list[int]) -> int:
    return sum(values)

def average(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
""",
"""# JavaScript function
export function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

export function isEmpty(value) {
  return value == null || value.length === 0;
}
""",
"""# TypeScript API response
type User = { id: string; name: string };

export function toUser(data: Record<string, unknown>): User {
  return {
    id: String(data.id ?? ""),
    name: String(data.name ?? "")
  };
}
""",
"""# SQL query
SELECT id, email, created_at
FROM users
WHERE active = true
ORDER BY created_at DESC
LIMIT 100;
""",
"""# Bug fixing
# Bug: division by zero when values is empty.
def average(values):
    if not values:
        return 0.0
    return sum(values) / len(values)
""",
"""# Unit test
def test_add():
    assert add(2, 3) == 5

def test_average_empty():
    assert average([]) == 0.0
""",
"""# HTTP handler
async function handler(request) {
  if (request.method !== "GET") {
    return new Response("Method Not Allowed", { status: 405 });
  }
  return Response.json({ ok: true });
}
""",
"""# Git workflow
git checkout -b feature/auth
git add .
git commit -m "Add authentication"
git push -u origin feature/auth
"""
]

def build(path, repeats):
    text = "\n\n<|endoftext|>\n\n".join(EXAMPLES * repeats)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")

if __name__ == "__main__":
    build("data/train.txt", 500)
    build("data/eval.txt", 20)
    print("Created bootstrap coding corpus in data/. Replace it with licensed real-world data before serious training.")
