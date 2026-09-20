import json
import jsonschema


def run_agent(question, tools, max_steps=10):
    # 建立 name -> tool 的映射表，方便快速查找
    tool_map = {t["name"]: t for t in tools}
    messages = [{"role": "user", "content": question}]

    last_sig = None  # 记录上一次调用的签名: (name, args_str)
    repeat_count = 0  # 连续重复计数

    for _ in range(max_steps):
        # 1. 调模型
        resp = llm.chat(messages=messages, tools=tools)

        # 没有工具调用，说明给出了最终答案，直接返回
        if not resp.get("tool_calls"):
            messages.append(resp.get("content", ""))  # 存入 assistant 消息，作用是记录模型回复，方便后续分析，但是这里直接返回模型回复，所以不需要记录
            return resp.get("content", "")  # 直接返回模型回复
        #messages.append(resp)  # 存入 assistant 消息，作用是记录模型回复，方便后续分析

        # 2. 执行工具
        for call in resp["tool_calls"]:
            name = call["function"]["name"]
            raw_args = call["function"]["arguments"]
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args # 解析 JSON 字符串为字典

            # 3. 连续重复调用熔断检测，sort_keys=True 确保参数顺序一致
            sig = (name, json.dumps(args, sort_keys=True)) #name和参数的组合作为签名，调用工具和参数相同才叫重复
            repeat_count = (repeat_count + 1) if sig == last_sig else 1
            last_sig = sig
            if repeat_count >= 3:
                return (
                    f"Loop terminated: '{name}' called 3 times with same args."
                )

            # 4. JSONSchema 校验与工具执行（异常转为 Observation）
            try:
                tool = tool_map[name]
                if "parameters" in tool:
                    jsonschema.validate(
                        instance=args, schema=tool["parameters"]
                    )
                obs = str(tool["func"](**args)) # 执行工具函数，将结果转换为字符串，这里是模拟工具函数的执行
            except Exception as e:
                obs = f"Execution/Validation Error: {e}"

            # 5. 回传 Observation
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "name": name,
                    "content": obs,
                }
            )

    return "Max steps reached without final answer."
    