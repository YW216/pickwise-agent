import json

from openai import OpenAI

from app.config.settings import settings

client=OpenAI(
    api_key=settings.openai_api_key,
    base_url=settings.openai_base_url,
)

tools=[
    {
        "id": "ecom_search",
        "type": "function",
        "function": {
            "name": "ecom_search",
            "description": "搜索商品",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "要搜索的商品名称"
                    }
                }
            }
        }
    }
]

message={
    "role": "user",
    "content": "搜索一个商品，商品名称为：商品A"
}
messages=[message]

for i in range(10):

    response=client.chat.completions.create(
    model=settings.model_name,
    messages=messages,
    temperature=settings.temperature,
    tools=tools
    )
    assistant_msg = response.choices[0].message
    print(f"第{i}次请求结果：")
    print("response:", response)
    print("response.choices[0]:", response.choices[0])
    print("response.choices[0].message:", assistant_msg)
    print("response.choices[0].message.content:", assistant_msg.content)

    if assistant_msg.tool_calls==None:
        print("结果：", assistant_msg.content)
        break

    # 追加 assistant 消息（含 tool_calls）
    messages.append({
        "role": "assistant",
        "content": assistant_msg.content,
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                },
            }
            for tc in assistant_msg.tool_calls
        ],
    })

    # 执行工具，并把结果作为 tool 消息回传
    for tc in assistant_msg.tool_calls:
        func_name = tc.function.name
        func_args = json.loads(tc.function.arguments)
        if func_name == "ecom_search":
            result = {"found": True, "name": func_args.get("query"), "price": "99.9 元", "stock": 10}
        else:
            result = {"error": f"未知工具 {func_name}"}
        result_str = json.dumps(result, ensure_ascii=False)
        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": result_str,
        })

