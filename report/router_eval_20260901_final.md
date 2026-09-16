========================================================================
Router 专业评测报告
用例集: router_cases.json（57 条）
模型: deepseek-v4-flash | 时间: 2026-09-01 19:16:17
========================================================================

## 一、逐条明细
[PASS] R001 [single_guide/easy] 明确求推荐 
      输入: '有什么适合写代码的笔记本推荐吗' → 期望 ['guide'] 实际 ['guide']
[PASS] R002 [single_guide/easy] 求推荐+预算 
      输入: '预算6000左右买什么手机' → 期望 ['guide'] 实际 ['guide']
[PASS] R003 [single_compare/easy] 两款对比 
      输入: '星海凌霄 Pro 14 和曜石磐石 15 哪个好' → 期望 ['compare'] 实际 ['compare']
[PASS] R004 [single_compare/easy] 怎么选 
      输入: '云章轻羽 Air 和墨白素笺 Earbuds 怎么选' → 期望 ['compare'] 实际 ['compare']
[PASS] R005 [single_consult/easy] 知识问题 
      输入: 'OLED 和 IPS 屏有什么区别' → 期望 ['consult'] 实际 ['consult']
[PASS] R006 [single_consult/easy] 政策问题 
      输入: '耳机一般保修几年' → 期望 ['consult'] 实际 ['consult']
[PASS] R007 [multi/easy] 推荐+知识双诉求 
      输入: '推荐个笔记本，顺便讲讲 OLED 和 IPS 的区别' → 期望 ['guide', 'consult'] 实际 ['guide', 'consult']
[PASS] R008 [multi/easy] 推荐+对比双诉求 
      输入: '帮我挑个耳机，顺便对比下凌霄 Buds Pro 和素笺 Earbuds' → 期望 ['guide', 'compare'] 实际 ['guide', 'compare']
[PASS] R009 [single_guide/medium] 隐含购买意图（无买字） 
      输入: '我想买个轻薄本' → 期望 ['guide'] 实际 ['guide']
[PASS] R010 [single_guide/medium] 收藏挑选（隐含） 
      输入: '从我的收藏里挑一款' → 期望 ['guide'] 实际 ['guide']
[PASS] R011 [single_consult/medium] 闲聊兜底 
      输入: '你好' → 期望 ['consult'] 实际 ['consult']
[PASS] R012 [single_consult/medium] 单款快查（无历史兜底） 
      输入: '这款多少钱' → 期望 ['consult'] 实际 ['consult']
[PASS] R013 [single_guide/hard] 需求描述无购买动词 
      输入: '写代码用，续航要好' → 期望 ['guide'] 实际 ['guide']
[PASS] R014 [single_guide/hard] 可行性咨询归 guide（2026-09-01 裁定：含预算+品类隐含选购诉求） 
      输入: '500 元预算能买到降噪耳机吗' → 期望 ['guide'] 实际 ['guide']
[PASS] R015 [multi/hard] 多意图隐含 
      输入: '选个手机，讲讲 OLED 屏伤不伤眼' → 期望 ['guide', 'consult'] 实际 ['guide', 'consult']
[PASS] R016 [continue/easy] 补预算（澄清延续） （history 2 条）
      输入: '500' → 期望 ['guide'] 实际 ['guide']
[PASS] R017 [continue/easy] 补用途 （history 2 条）
      输入: '日常办公' → 期望 ['guide'] 实际 ['guide']
[PASS] R018 [continue/easy] 补用途 （history 2 条）
      输入: '打游戏' → 期望 ['guide'] 实际 ['guide']
[PASS] R019 [continue/medium] 真实漂移场景复现（两轮+建议混合消息） （history 4 条）
      输入: '600' → 期望 ['guide'] 实际 ['guide']
[PASS] R020 [continue/medium] 补尺寸偏好 （history 2 条）
      输入: '14寸的' → 期望 ['guide'] 实际 ['guide']
[PASS] R021 [continue/medium] 复合补充信息 （history 2 条）
      输入: '预算500，主要通勤' → 期望 ['guide'] 实际 ['guide']
[PASS] R022 [continue/hard] 模糊回应延续 （history 2 条）
      输入: '随便，你看着办' → 期望 ['guide'] 实际 ['guide']
[PASS] R023 [continue/hard] 澄清中插入知识问题（双诉求） （history 2 条）
      输入: '那 OLED 屏伤眼吗' → 期望 ['guide', 'consult'] 实际 ['guide', 'consult']
[PASS] R024 [switch/easy] 推荐后问政策（换场景） （history 2 条）
      输入: '这款保修几年' → 期望 ['consult'] 实际 ['consult']
[PASS] R025 [switch/easy] 推荐后问单款参数（规则4归 consult） （history 2 条）
      输入: '那续航呢' → 期望 ['consult'] 实际 ['consult']
[PASS] R026 [switch/medium] 推荐后切比选 （history 2 条）
      输入: '那凌霄 Pro 14 和磐石 15 哪个好' → 期望 ['compare'] 实际 ['compare']
[PASS] R027 [switch/medium] 继续推荐流程（追问更便宜的） （history 2 条）
      输入: '还有更便宜的吗' → 期望 ['guide'] 实际 ['guide']
[PASS] R028 [switch/hard] 推荐后切知识问题 （history 2 条）
      输入: 'OLED 和 IPS 屏有什么区别' → 期望 ['consult'] 实际 ['consult']
[PASS] R029 [switch/hard] 咨询后切推荐 （history 2 条）
      输入: '推荐一款护眼的笔记本吧' → 期望 ['guide'] 实际 ['guide']
[PASS] R030 [boundary/easy] 致谢兜底 
      输入: '谢谢' → 期望 ['consult'] 实际 ['consult']
[PASS] R031 [boundary/medium] 无历史短消息（无法分类兜底） 
      输入: '贵吗' → 期望 ['consult'] 实际 ['consult']
[PASS] R032 [boundary/medium] 无历史纯数字（无法分类兜底） 
      输入: '500' → 期望 ['consult'] 实际 ['consult']
[PASS] R033 [boundary/hard] 无历史指代不明（隐含对比意图） 
      输入: '这两款差在哪' → 期望 ['compare'] 实际 ['compare']
[PASS] R034 [boundary/hard] 空输入/无意义输入（兜底） 
      输入: '，，，' → 期望 ['consult'] 实际 ['consult']
[PASS] R035 [continue/medium] 多轮澄清递进：补用途（第2轮） （history 4 条）
      输入: '运动' → 期望 ['guide'] 实际 ['guide']
[PASS] R036 [continue/medium] 多轮澄清递进：补重量偏好（第2轮） （history 4 条）
      输入: '轻一点的' → 期望 ['guide'] 实际 ['guide']
[PASS] R037 [continue/easy] 补品牌偏好（澄清延续） （history 2 条）
      输入: '不要苹果生态的' → 期望 ['guide'] 实际 ['guide']
[PASS] R038 [continue/easy] 补连接方式偏好 （history 2 条）
      输入: '有线耳机' → 期望 ['guide'] 实际 ['guide']
[PASS] R039 [continue/easy] 手机品类澄清延续 （history 2 条）
      输入: '4000 左右' → 期望 ['guide'] 实际 ['guide']
[PASS] R040 [continue/easy] 笔记本品类澄清延续 （history 2 条）
      输入: '写代码为主' → 期望 ['guide'] 实际 ['guide']
[PASS] R041 [continue/medium] 比选场景澄清延续：补偏好维度 （history 2 条）
      输入: '续航更重要' → 期望 ['compare'] 实际 ['compare']
[PASS] R042 [continue/medium] 比选场景澄清延续：补选择倾向 （history 2 条）
      输入: '便宜的那款' → 期望 ['compare'] 实际 ['compare']
[PASS] R043 [continue/medium] 澄清后一次性补全完整需求 （history 2 条）
      输入: '预算5000，主要办公，最好轻一点' → 期望 ['guide'] 实际 ['guide']
[PASS] R044 [continue/hard] 三轮澄清递进（预算→用途→偏好） （history 6 条）
      输入: '轻一点的' → 期望 ['guide'] 实际 ['guide']
[PASS] R045 [continue/medium] 澄清中换品类（仍属选购） （history 2 条）
      输入: '算了，我还是看看手机吧' → 期望 ['guide'] 实际 ['guide']
[PASS] R046 [continue/medium] 澄清中终止购买（非选购，兜底） （history 2 条）
      输入: '算了，先不买了' → 期望 ['consult'] 实际 ['consult']
[PASS] R047 [continue/hard] 澄清中模糊反问（延续） （history 4 条）
      输入: '你看着推荐吧' → 期望 ['guide'] 实际 ['guide']
[PASS] R048 [continue/hard] 澄清中锁定候选跳澄清（转比选） （history 2 条）
      输入: '那凌霄 Pro 14 和磐石 15 哪个好' → 期望 ['compare'] 实际 ['compare']
[PASS] R049 [multi/easy] 推荐+政策双诉求 
      输入: '推荐个耳机，顺便问下保修多久' → 期望 ['guide', 'consult'] 实际 ['guide', 'consult']
[PASS] R050 [multi/medium] 比选+知识双诉求 
      输入: '凌霄 Buds Pro 和素笺 Earbuds 哪个好，顺便讲讲降噪原理' → 期望 ['compare', 'consult'] 实际 ['compare', 'consult']
[PASS] R051 [multi/medium] 比选+政策双诉求 
      输入: '凌霄 Pro 14 和磐石 15 差在哪，保修有什么不同' → 期望 ['compare', 'consult'] 实际 ['compare', 'consult']
[PASS] R052 [multi/hard] 三意图（推荐+对比+知识） 
      输入: '帮我选个耳机，对比下凌霄和素笺，顺便讲讲什么是 LDAC' → 期望 ['guide', 'compare', 'consult'] 实际 ['guide', 'compare', 'consult']
[PASS] R053 [multi/hard] 收藏+知识双诉求（无显式连接词） 
      输入: '从我的收藏里挑个游戏耳机，顺便讲讲低延迟的原理' → 期望 ['guide', 'consult'] 实际 ['guide', 'consult']
[PASS] R054 [multi/medium] 推荐+比选混合（无顺便连接词） 
      输入: '推荐个 6000 内的笔记本，星海凌霄和曜石磐石我该选哪个' → 期望 ['guide', 'compare'] 实际 ['guide', 'compare']
[PASS] R055 [multi/hard] 比选+知识（无连接词，隐含双诉求） 
      输入: '这两款哪个更适合编程，OLED 屏会不会伤眼' → 期望 ['compare', 'consult'] 实际 ['compare', 'consult']
[PASS] R056 [boundary/hard] 跨压缩指代（无历史，靠 summary 兜底回选购） 
      输入: '我上次问的那个耳机' → 期望 ['guide'] 实际 ['guide']
[PASS] R057 [boundary/hard] 跨压缩指代（已知限制：纯指示代词无关键词，summary 无法解析，兜底 consult） 
      输入: '就我之前说的那款' → 期望 ['consult'] 实际 ['consult']

## 二、汇总
完全匹配准确率: 57/57 = 100.0%

## 三、场景级指标（多标签：精确率 / 召回率 / F1）
  场景         召回(exp→act)      精确(act→exp)      F1    
  guide      100%             100%             1.00    (34 例)
  compare    100%             100%             1.00    (13 例)
  consult    100%             100%             1.00    (22 例)

## 四、分维度（group）
  boundary         7/7 = 100%
  continue         22/22 = 100%
  multi            10/10 = 100%
  single_compare   2/2 = 100%
  single_consult   4/4 = 100%
  single_guide     6/6 = 100%
  switch           6/6 = 100%

## 五、分难度（difficulty）
  easy       19/19 = 100%
  hard       17/17 = 100%
  medium     21/21 = 100%

## 六、失败分析（0 条）