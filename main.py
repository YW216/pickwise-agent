
from app.multi_agent.orchestrator import MultiAgentOrchestrator


def main():
    agent = MultiAgentOrchestrator()

    print("=" * 50)
    print("  PickWise · 选购助手「小P」(Multi-Agent 协作模式)")
    print("  支持工具调用 + 知识检索 + 用户记忆 + 技能编排")
    print("  输入 quit/exit 退出, reset 重置, memory 查看记忆, skills 查看技能")
    print("=" * 50)
    print()

    if agent.history_size > 0:
        print(f"💬 已恢复上次对话（{agent.history_size} 条历史）\n")

    while True:
        try:
            user_input = input("👤 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            agent.save()
            agent.close()
            print("\n再见，欢迎下次光临！")
            break

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit"):
            agent.save()
            agent.close()
            print("再见，欢迎下次光临！")
            break

        if user_input.lower() == "reset":
            agent.reset()
            print("对话已重置。\n")
            continue

        if user_input.lower() == "skills":
            if hasattr(agent, "skill_manager") and agent.skill_manager.enabled:
                catalog = agent.skill_manager.get_catalog()
                print(f"\n--- 已加载 {len(catalog)} 个技能 ---")
                for s in catalog:
                    print(f"  - {s['name']}：{s['description']}")
                print()
            else:
                print("技能系统未启用\n")
            continue

        if user_input.lower() == "memory":
            if hasattr(agent, "memory_manager") and agent.memory_manager.memory_enabled:
                stm = agent.memory_manager.stm
                ltm = agent.memory_manager.ltm
                print("\n--- 短期记忆（本次对话）---")
                if stm.facts:
                    for f in stm.facts:
                        print(f"  - {f}")
                else:
                    print("  （暂无）")
                print(f"\n--- 长期记忆（跨会话，用户: {ltm.user_id}）---")
                if ltm.facts:
                    for f in ltm.facts:
                        print(f"  - [{f.category}] {f.content}")
                else:
                    print("  （暂无）")
                if ltm.interaction_summaries:
                    print("\n--- 最近交互 ---")
                    for s in ltm.interaction_summaries[-3:]:
                        print(f"  - {s['summary']}")
                print()
            else:
                print("记忆功能未启用\n")
            continue

        try:
            # chat() 返回 reply 文本（设计文档十一节，P1 起）
            response = agent.chat(user_input)
            # 分隔线划清"过程流（路由/思考/工具）"与"最终结果"的边界
            print("\n" + "─" * 50)
            print(f"🤖 小P最终回复: {response}\n")

        except Exception as e:
            print(f"\n⚠️  出错了: {e}\n")


if __name__ == "__main__":
    main()
