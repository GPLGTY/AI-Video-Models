# H3-lite-Te 提示词模板：参考图模式 / 首尾帧模式

适用服务：`H3L`（TeSpeed），`mode=ref2va` / `mode=fl2va`。  
结构对齐官方 skill：`MiniMax-H3/skills/h3-prompt-writing`。

| 模式 | API | 参考图含义 | 提示词重写 |
|------|-----|------------|------------|
| 参考图 | `ref2va` | `<Picture N>` 提供人物/场景/道具身份，**不是**首尾关键帧 | 当前**不**自动重写，建议按模板写全 |
| 首尾帧 / 仅首帧 | `fl2va` | 有末帧：Picture1=开场、Picture2=结束；仅首帧：Picture1=开场 | **默认会**经 Gemma 重写（首+尾→FL2VA；仅首帧→I2VA）；业务侧可先交中文简版 |

台词/口播必须原样保留（改写后进 `<d>[语言] …</d>`）。画面描述可中英，**标准送模结构推荐英文字段名**。

---

## 一、首尾帧模式（FL2VA）

### 1.1 业务简版（中文填空 → 交给网关自动重写）

把 `【】` 换成具体内容；时长 `S.SS` 与请求里的 `duration_seconds` 一致。

```text
【任务】根据首帧与尾帧生成 {S.SS} 秒短视频；首帧=开场，尾帧=结束态。
【画幅】{9:16 / 16:9 / …}；主体在安全区内；无边框、拼图、水印。

【首帧状态】（对应 first_frame）
人物/主体：{身份、服装、位置、表情、手持物}
场景与光线：{环境、主光方向}
构图与景别：{中近景/全景等}

【动作路径】（重点：怎么从首帧走到尾帧，不要只复述两张静图）
0.00s 起：{开场动作}
过程：{姿态变化、道具操作、表情变化、镜头是否固定/轻推/手持微晃}
结束前：{逐步收敛到尾帧构图与姿势}

【尾帧状态】（对应 last_frame）
必须落到：{姿态、手位、道具位置、构图}
禁止：淡出、黑屏、收手收表情（除非尾帧本身如此）

【声音】
口播/对白（原语言原句，勿改写）：
« {台词全文} »
环境音：{室内底噪 / 车外车流 / …}
背景音乐：{无 / 轻钢琴慢速 / …}

【约束】
- 人物五官、服装、场景与首尾帧一致，中途不漂移
- 优先单镜头连续插值；除非明确要求，不要硬切多镜
- 禁止：{无关人物、Logo、字幕贴纸、夸张电影运镜…}
```

### 1.2 标准结构（关闭重写或人工终稿时使用）

首行指令固定；`S.SS` = 时长两位小数；`Shot N` 为实际最后一镜编号（单镜任务写 Shot 1）。

```text
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot N) aligns with the S.SS-second mark of the target video.

integrated_multimodal_description: [Shot 1] {style}, {opening composition matching Picture 1}. {camera: fixed / push in small amplitude slow / handheld micro-shake}. {subject action path}. {speaker (S1) says: <d>[Chinese/English] …</d>}. Toward the end, pose, spacing, props, and framing converge to Picture 2.

overall_soundscape: {ambience + physical action sounds across the full clip; no dialogue here}.

non_diegetic_music: {instrumentation + tempo + dynamics} | N/A
```

### 1.3 填好的短例（8 秒单镜）

```text
How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the 0.00-second mark of the target video; Picture 2 (from Shot 1) aligns with the 8.00-second mark of the target video.

integrated_multimodal_description: [Shot 1] Live-action, UGC phone selfie, a medium close-up begins from Picture 1: a young woman in a light yellow T-shirt stands before pale tiled wall, both hands mid-wipe near her cheeks. The camera stays mostly fixed with slight handheld shake as she slows the wipe, opens her palms beside her cheeks in a soft pinch-and-show gesture, and settles into the smile, hand spacing, and framing of Picture 2. She (S1) says: <d>[Chinese] 你看，皮肤状态好多了。</d>

overall_soundscape: Quiet indoor room tone, soft fabric rustle from sleeves, faint finger-skin contact sounds.

non_diegetic_music: N/A
```

**写法要点**

- 写「首帧 → 中间变化 → 尾帧」，少写两段静态图描述。
- 默认单镜；多镜时用 `[Shot 2] At 00:03.500, the camera cuts to...`。
- 运镜三要素：类型 + 幅度 + 速度（中等可省略）。

---

## 二、参考图模式（Ref2VA）

参考图用于**身份/外观/场景结构**，成片时间轴由提示词自己写。图序与请求 `ref_images` 一致：`<Picture 1>` = 第 1 张，以此类推。

### 2.1 业务简版（中文填空，适合 UGC / 口播）

```text
生成 {时长} 秒 {竖屏9:16/横屏16:9} {超写实真人 / 电影感 / …} 视频。

【参考素材】
<Picture 1> → {人物五官、发型、服装；仅取长相不取画风（若参考是插画需写死）}
<Picture 2> → {场景结构 / 产品外观 / …}
（按需继续 Picture 3…）

【固定人物】
{年龄、种族、发型、服装、配饰、妆造}
全片五官/服装/发型一致，不漂移。
画风：{实拍手机 / cinematic}；禁止沿用参考图的绘画笔触（如有）。

【固定场景与光线】
{地点、主光、是否补光、背景可见元素}

【固定机位】
{自拍固定 / 三脚架 / 手持微晃}
禁止：{轨道推拉、多机位乱切、慢动作…}（按需）

【声音】
音色：{年龄口音语速}
口播原文（勿改写）：
« {逐句台词} »
环境音：{…}
BGM：{无 / …}

【分镜时间轴】
【0:00–0:0X】镜1
景别/构图：{…}
动作：{…}
台词：« … »
防穿模/手部：{双手数量、勿穿入道具…}

【0:0X–0:0Y】镜2
…

【结尾】
最后一帧必须：{姿势/道具状态}
禁止：淡出、黑屏、字幕贴纸、水印、无关 Logo。
```

### 2.2 标准六段结构（对齐 Ref2VA skill，英文字段）

```text
subject_definitions:
<Subject 1> is the {person/object} whose appearance comes from <Picture 1>, {key visual traits}.
<Subject 2> is the {environment/product} from <Picture 2>, {what to keep}.
（仅当某图本身作关键帧/构图锚点时才单独写 <Picture N> 行；纯身份参考写进 Subject 即可。）

summary:
[reference generation] The target video is a {duration}-second {style} clip in which <Subject 1> performs {main action} in <Subject 2>, guided by <Picture 1>/<Picture 2> for identity and scene structure.

retention_analysis:
<Subject 1> (appears in [Shot 1], …): fully_preserved - face, hair, outfit stay consistent; new actions are allowed.
<Subject 2> (appears in [Shot 1], …): fully_preserved - layout/materials kept; lighting follows the shot plan.
（可选）<Picture 1>: weak_reference - identity source only, not used as a literal first frame.

detailed_description:
The target video is in a {style} look with {lighting}.
[Shot 1] Medium shot of <Subject 1> in <Subject 2>... {camera}. <Subject 1> (S1) says: <d>[English] ...</d>
[Shot 2] At 00:04.000, the camera cuts to...

overall_soundscape: {ambience + foley across the full video}.

non_diegetic_music: {score} | N/A
```

**关系词速查（retention_analysis）**

| 标记 | 含义 |
|------|------|
| `fully_preserved` | 定义特征完整保留 |
| `partially_preserved` | 仍用该参考，但部分特征改动 |
| `attribute_transfer` | 特征迁移到另一主体 |
| `weak_reference` | 仅风格/品类/氛围相似 |

### 2.3 最小英文骨架（人物+场景两张图）

```text
subject_definitions:
<Subject 1> is the young woman in <Picture 1>, with her exact facial features, hairstyle, and outfit.
<Subject 2> is the indoor balcony setting from <Picture 2>, including the glass door and warm wood column.

summary:
[reference generation] A 10.00-second vertical UGC selfie where <Subject 1> talks to camera at a table in <Subject 2> while handling food props.

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2]): fully_preserved - identity and wardrobe locked; performance and gestures follow the timeline.
<Subject 2> (appears in [Shot 1], [Shot 2]): fully_preserved - architecture and daylight direction retained.

detailed_description:
The target video is live-action phone UGC with soft daylight and mild handheld shake.
[Shot 1] Medium close-up of <Subject 1> centered at the table in <Subject 2>... She (S1) says: <d>[English] ...</d>
[Shot 2] At 00:03.000, the framing widens slightly as she points at the container...

overall_soundscape: Soft outdoor ambience beyond the glass, light table bumps, fabric rustle.

non_diegetic_music: N/A
```

---

## 三、共用约定

1. **标签编号**：与上传顺序一致；Ref2VA 呈现顺序一般为 图 → 视频 → 独立音频。  
2. **对白**：原文进 `<d>[Chinese] …</d>` / `<d>[English] …</d>`，不改词、不换序。  
3. **时长**：FL2VA 指令行里的 `S.SS`、分镜切点，必须落在 `duration_seconds` 内。  
4. **TeSpeed**：默认开；与提示词模板无关。Ref2VA 可另开 `use_realism_lora`（请求显式打开）。  
5. **FL2VA 关重写**：请求加 `"prompt_rewrite": false`，或环境变量 `H3L_FL2VA_PROMPT_REWRITE=0`。

### 请求字段对照

```json
{
  "mode": "fl2va",
  "input": {
    "prompt": "<按第一节模板>",
    "first_frame": "https://.../first.png",
    "last_frame": "https://.../last.png",
    "duration_seconds": 8,
    "short_edge": 720
  }
}
```

```json
{
  "mode": "ref2va",
  "input": {
    "prompt": "<按第二节模板>",
    "ref_images": ["https://.../person.png", "https://.../scene.png"],
    "duration_seconds": 10,
    "aspect_ratio": "9:16",
    "short_edge": 768
  }
}
```

更多 API 说明见 `docs/DEPLOY.md`。
