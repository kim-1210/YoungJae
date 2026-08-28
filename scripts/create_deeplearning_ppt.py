from __future__ import annotations

import math
import textwrap
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "presentation_assets"
PPTX_PATH = ROOT / "deep_learning_explanation.pptx"

SLIDE_W_IN = 13.333
SLIDE_H_IN = 7.5
DPI = 160
SLIDE_W_EMU = 12192000
SLIDE_H_EMU = 6858000


def setup_font() -> None:
    preferred = [
        "Noto Sans CJK KR",
        "NanumGothic",
        "Malgun Gothic",
        "AppleGothic",
        "DejaVu Sans",
    ]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in preferred:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def add_wrapped(ax, text, x, y, width=40, fontsize=24, color="#111827", weight="normal", lineheight=1.22):
    lines = []
    for part in text.split("\n"):
        if part.strip() == "":
            lines.append("")
        else:
            lines.extend(textwrap.wrap(part, width=width))
    ax.text(
        x,
        y,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        color=color,
        weight=weight,
        linespacing=lineheight,
    )


def draw_title(ax, title, subtitle=None):
    ax.text(0.06, 0.9, title, transform=ax.transAxes, ha="left", va="top", fontsize=34, weight="bold", color="#111827")
    if subtitle:
        add_wrapped(ax, subtitle, 0.06, 0.8, width=62, fontsize=18, color="#4b5563")


def rounded_box(ax, xy, wh, title, body="", fc="#f4f8ff", ec="#263238", title_size=21, body_size=15):
    x, y = xy
    w, h = wh
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=2,
        edgecolor=ec,
        facecolor=fc,
        transform=ax.transAxes,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h * 0.62, title, transform=ax.transAxes, ha="center", va="center", fontsize=title_size, weight="bold", color="#111827")
    if body:
        ax.text(x + w / 2, y + h * 0.34, body, transform=ax.transAxes, ha="center", va="center", fontsize=body_size, color="#374151", linespacing=1.25)


def arrow(ax, start, end):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops=dict(arrowstyle="->", lw=2.8, color="#111827"),
    )


def bullet_list(ax, items, x, y, width=44, fontsize=20):
    current = y
    for item in items:
        ax.text(x, current, "•", transform=ax.transAxes, ha="left", va="top", fontsize=fontsize + 3, color="#2563eb", weight="bold")
        add_wrapped(ax, item, x + 0.035, current, width=width, fontsize=fontsize, color="#111827")
        current -= 0.105 + 0.028 * max(0, len(textwrap.wrap(item, width=width)) - 1)


def make_canvas():
    fig, ax = plt.subplots(figsize=(SLIDE_W_IN, SLIDE_H_IN), dpi=DPI)
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def save_slide(fig, number: int) -> Path:
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"slide_{number:02d}.png"
    fig.savefig(path, dpi=DPI, bbox_inches=None, facecolor="white")
    plt.close(fig)
    return path


def slide_01():
    fig, ax = make_canvas()
    ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, facecolor="#f7fbff", edgecolor="none"))
    ax.text(0.06, 0.76, "딥러닝 핵심 개념", transform=ax.transAxes, fontsize=46, weight="bold", color="#111827")
    ax.text(0.06, 0.64, "MNIST 숫자 인식 코드에 나오는 단어들을 쉽게 이해하기", transform=ax.transAxes, fontsize=22, color="#374151")
    rounded_box(ax, (0.07, 0.22), (0.2, 0.22), "그림", "손글씨 숫자", "#eef6ff")
    rounded_box(ax, (0.39, 0.22), (0.2, 0.22), "모델", "단서 찾기", "#edf9f0")
    rounded_box(ax, (0.71, 0.22), (0.2, 0.22), "정답", "0~9 중 하나", "#fbeff5")
    arrow(ax, (0.29, 0.33), (0.37, 0.33))
    arrow(ax, (0.61, 0.33), (0.69, 0.33))
    return save_slide(fig, 1)


def slide_02():
    fig, ax = make_canvas()
    draw_title(ax, "Deep Learning이 뭐예요?", "컴퓨터가 많은 예시를 보면서 스스로 규칙을 조금씩 찾아가는 방법입니다.")
    rounded_box(ax, (0.07, 0.45), (0.2, 0.2), "예시", "숫자 그림\n많이 보기", "#eef6ff")
    rounded_box(ax, (0.4, 0.45), (0.2, 0.2), "연습", "예측하고\n틀린 만큼 고치기", "#fff7e6")
    rounded_box(ax, (0.73, 0.45), (0.2, 0.2), "실력", "처음 보는 숫자도\n맞히기", "#edf9f0")
    arrow(ax, (0.29, 0.55), (0.38, 0.55))
    arrow(ax, (0.62, 0.55), (0.71, 0.55))
    bullet_list(
        ax,
        [
            "사람이 모든 규칙을 직접 알려주는 것이 아닙니다.",
            "모델이 여러 층을 지나며 그림 속 단서를 찾습니다.",
            "많이 틀리면 많이 고치고, 조금 틀리면 조금 고칩니다.",
        ],
        0.08,
        0.26,
        width=58,
        fontsize=19,
    )
    return save_slide(fig, 2)


def slide_03():
    fig, ax = make_canvas()
    draw_title(ax, "MNIST 모델의 전체 흐름", "손글씨 이미지를 숫자 줄로 바꾸고, 단서를 찾아서 답을 고릅니다.")
    labels = [
        ("손글씨 그림", "28 x 28칸"),
        ("숫자 줄", "784개 값"),
        ("단서 찾기", "선, 꺾임, 둥근 모양"),
        ("답 고르기", "0~9 점수 비교"),
    ]
    xs = [0.06, 0.31, 0.56, 0.81]
    colors = ["#eef6ff", "#fff7e6", "#edf9f0", "#fbeff5"]
    for x, (title, body), color in zip(xs, labels, colors):
        rounded_box(ax, (x, 0.4), (0.16, 0.24), title, body, color, title_size=19, body_size=14)
    for x in [0.235, 0.485, 0.735]:
        arrow(ax, (x, 0.52), (x + 0.055, 0.52))
    add_wrapped(ax, "중요한 생각: 모델은 이미지를 사람처럼 보는 것이 아니라, 숫자들을 계산해서 가장 그럴듯한 답을 고릅니다.", 0.1, 0.22, width=58, fontsize=22, color="#111827")
    return save_slide(fig, 3)


def slide_04():
    fig, ax = make_canvas()
    draw_title(ax, "Linear는 무엇인가요?", "여러 숫자를 섞어서 새로운 숫자를 만드는 계산입니다.")
    ax.text(0.08, 0.58, "입력 숫자들", transform=ax.transAxes, fontsize=22, weight="bold")
    ax.text(0.43, 0.58, "중요도 곱하기", transform=ax.transAxes, fontsize=22, weight="bold")
    ax.text(0.75, 0.58, "새 결과", transform=ax.transAxes, fontsize=22, weight="bold")
    for i, val in enumerate(["0.1", "0.8", "0.3", "0.0"]):
        rounded_box(ax, (0.09, 0.46 - i * 0.07), (0.11, 0.052), val, "", "#eef6ff", title_size=16)
    ax.text(0.34, 0.33, "x 가중치 + 더하기", transform=ax.transAxes, fontsize=25, weight="bold", color="#2563eb")
    rounded_box(ax, (0.75, 0.35), (0.14, 0.12), "특징 점수", "예: 둥근 정도", "#edf9f0", title_size=18, body_size=13)
    arrow(ax, (0.23, 0.42), (0.33, 0.39))
    arrow(ax, (0.6, 0.39), (0.73, 0.41))
    add_wrapped(
        ax,
        "`nn.Linear(784, 512)`는 784개 숫자를 보고 512개의 새 단서 점수를 만듭니다.\n학습은 이 가중치를 더 좋은 값으로 고치는 과정입니다.",
        0.08,
        0.15,
        width=58,
        fontsize=18,
        color="#111827",
    )
    return save_slide(fig, 4)


def slide_05():
    fig, ax = make_canvas()
    draw_title(ax, "sigmoid는 무엇인가요?", "큰 숫자나 작은 숫자를 0과 1 사이로 눌러주는 함수입니다.")
    x = np.linspace(-8, 8, 400)
    y = 1 / (1 + np.exp(-x))
    plot_ax = fig.add_axes([0.08, 0.20, 0.42, 0.38])
    plot_ax.plot(x, y, color="#2563eb", lw=4)
    plot_ax.axhline(0, color="#9ca3af")
    plot_ax.axhline(1, color="#9ca3af")
    plot_ax.axvline(0, color="#9ca3af", lw=1)
    plot_ax.set_ylim(-0.05, 1.05)
    plot_ax.grid(True, alpha=0.25)
    bullet_list(
        ax,
        [
            "계산 결과를 너무 크거나 작지 않게 정리합니다.",
            "0에 가까우면 약한 신호,\n1에 가까우면 강한 신호처럼 볼 수 있습니다.",
            "모델 중간에서 단서를\n부드럽게 바꿔주는 역할을 합니다.",
        ],
        0.58,
        0.62,
        width=22,
        fontsize=17,
    )
    return save_slide(fig, 5)


def slide_06():
    fig, ax = make_canvas()
    draw_title(ax, "log_softmax는 무엇인가요?", "10개의 점수를 비교해서 어떤 숫자가 가장 그럴듯한지 보기 쉽게 바꿉니다.")
    nums = list(range(10))
    scores = np.array([0.2, 0.5, 1.0, 0.1, 0.4, 4.0, 0.6, 0.3, 0.9, 0.2])
    probs = np.exp(scores) / np.exp(scores).sum()
    bar_ax = fig.add_axes([0.08, 0.20, 0.48, 0.38])
    colors = ["#bfdbfe"] * 10
    colors[5] = "#2563eb"
    bar_ax.bar(nums, probs, color=colors)
    bar_ax.set_xticks(nums)
    bar_ax.set_ylim(0, 1)
    bar_ax.set_xlabel("숫자")
    bar_ax.set_ylabel("가능성")
    bullet_list(
        ax,
        [
            "모델 마지막에는 숫자 0~9에 대한\n점수 10개가 나옵니다.",
            "`softmax`는 점수를 가능성처럼\n비교하게 해줍니다.",
            "`log`는 손실 계산을 더 안정적으로\n하려고 붙습니다.",
            "가장 높은 점수를 가진 숫자가\n모델의 예측입니다.",
        ],
        0.61,
        0.62,
        width=23,
        fontsize=16,
    )
    return save_slide(fig, 6)


def slide_07():
    fig, ax = make_canvas()
    draw_title(ax, "CrossEntropyLoss는 무엇인가요?", "모델이 정답에 얼마나 자신 있게 가까워졌는지 점수로 알려줍니다.")
    rounded_box(ax, (0.08, 0.48), (0.22, 0.18), "정답", "숫자 5", "#eef6ff")
    rounded_box(ax, (0.39, 0.48), (0.22, 0.18), "모델 예측", "5일 가능성 80%", "#edf9f0")
    rounded_box(ax, (0.70, 0.48), (0.22, 0.18), "손실", "작음", "#fff7e6")
    arrow(ax, (0.32, 0.57), (0.37, 0.57))
    arrow(ax, (0.63, 0.57), (0.68, 0.57))
    rounded_box(ax, (0.39, 0.2), (0.22, 0.18), "틀린 예측", "5일 가능성 5%", "#fee2e2")
    rounded_box(ax, (0.70, 0.2), (0.22, 0.18), "손실", "큼", "#fee2e2")
    arrow(ax, (0.63, 0.29), (0.68, 0.29))
    add_wrapped(
        ax,
        "핵심: 정답을 자신 있게 맞히면 손실이 작아집니다.\n틀린 답을 자신 있게 말하면 손실이 커집니다.",
        0.08,
        0.12,
        width=48,
        fontsize=19,
    )
    return save_slide(fig, 7)


def slide_08():
    fig, ax = make_canvas()
    draw_title(ax, "SGD는 무엇인가요?", "틀린 만큼 모델의 가중치를 조금씩 고치는 방법입니다.")
    curve_ax = fig.add_axes([0.08, 0.20, 0.45, 0.38])
    x = np.linspace(-4, 4, 200)
    y = 0.18 * (x + 0.4) ** 2 + 0.15
    curve_ax.plot(x, y, lw=4, color="#2563eb")
    points = [3.2, 2.0, 1.0, 0.25, -0.35]
    curve_ax.scatter(points, [0.18 * (p + 0.4) ** 2 + 0.15 for p in points], s=80, color="#ef4444")
    for a, b in zip(points[:-1], points[1:]):
        curve_ax.annotate("", xy=(b, 0.18 * (b + 0.4) ** 2 + 0.15), xytext=(a, 0.18 * (a + 0.4) ** 2 + 0.15), arrowprops=dict(arrowstyle="->", lw=2))
    curve_ax.set_xlabel("가중치")
    curve_ax.set_ylabel("손실")
    curve_ax.grid(True, alpha=0.25)
    bullet_list(
        ax,
        [
            "`SGD`는 한 번에 완벽하게 고치지 않습니다.",
            "작은 묶음의 데이터를 보고\n조금씩 방향을 잡습니다.",
            "`lr=0.01`은 한 번에 얼마나\n움직일지 정하는 값입니다.",
            "`momentum`은 이전에 가던 방향을\n조금 기억해서 움직이는 느낌입니다.",
        ],
        0.6,
        0.62,
        width=23,
        fontsize=16,
    )
    return save_slide(fig, 8)


def slide_09():
    fig, ax = make_canvas()
    draw_title(ax, "학습은 이렇게 반복됩니다", "예측하고, 틀린 정도를 계산하고, 조금 고치는 일을 계속합니다.")
    steps = [
        ("1. 예측", "모델이 답을 고름", "#eef6ff"),
        ("2. 손실 계산", "얼마나 틀렸는지 봄", "#fff7e6"),
        ("3. 뒤로 계산", "어디를 고칠지 찾음", "#edf9f0"),
        ("4. 가중치 수정", "SGD로 조금 고침", "#fbeff5"),
    ]
    coords = [(0.12, 0.52), (0.58, 0.52), (0.58, 0.23), (0.12, 0.23)]
    for (title, body, color), coord in zip(steps, coords):
        rounded_box(ax, coord, (0.28, 0.18), title, body, color, title_size=22, body_size=15)
    arrow(ax, (0.41, 0.61), (0.56, 0.61))
    arrow(ax, (0.72, 0.5), (0.72, 0.43))
    arrow(ax, (0.56, 0.32), (0.41, 0.32))
    arrow(ax, (0.26, 0.43), (0.26, 0.5))
    add_wrapped(ax, "이 반복을 여러 번 하면 모델은 조금씩 더 좋은 답을 고르게 됩니다.", 0.16, 0.09, width=55, fontsize=23)
    return save_slide(fig, 9)


def slide_10():
    fig, ax = make_canvas()
    draw_title(ax, "한 장 정리", "코드 속 어려운 이름을 쉬운 말로 바꾸면 이렇게 볼 수 있습니다.")
    rows = [
        ("Deep Learning", "예시를 많이 보고 스스로 규칙을 찾는 방법"),
        ("Linear", "입력 숫자들을 섞어서 새 단서 점수를 만드는 계산"),
        ("sigmoid", "계산 결과를 0과 1 사이로 부드럽게 정리"),
        ("log_softmax", "0~9 점수를 비교하기 좋게 바꿈"),
        ("CrossEntropyLoss", "정답과 예측이 얼마나 다른지 알려주는 점수"),
        ("SGD", "손실을 줄이도록 가중치를 조금씩 고치는 방법"),
    ]
    y = 0.67
    for name, desc in rows:
        rounded_box(ax, (0.08, y - 0.03), (0.25, 0.07), name, "", "#eef6ff", title_size=17)
        ax.text(0.38, y + 0.005, desc, transform=ax.transAxes, ha="left", va="center", fontsize=20, color="#111827")
        y -= 0.095
    add_wrapped(ax, "외울 때는 이름보다 역할을 먼저 기억하면 훨씬 쉽습니다.", 0.16, 0.08, width=55, fontsize=23, color="#2563eb", weight="bold")
    return save_slide(fig, 10)


def xml_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>
</Relationships>"""


def content_types(slide_count: int) -> str:
    slide_overrides = "\n".join(
        f'  <Override PartName="/ppt/slides/slide{i}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
        for i in range(1, slide_count + 1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>
  <Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>
  <Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>
  <Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>
{slide_overrides}
</Types>"""


def presentation_xml(slide_count: int) -> str:
    sld_ids = "\n".join(
        f'    <p:sldId id="{255 + i}" r:id="rId{i}"/>'
        for i in range(1, slide_count + 1)
    )
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:sldMasterIdLst>
    <p:sldMasterId id="2147483648" r:id="rId{slide_count + 1}"/>
  </p:sldMasterIdLst>
  <p:sldIdLst>
{sld_ids}
  </p:sldIdLst>
  <p:sldSz cx="{SLIDE_W_EMU}" cy="{SLIDE_H_EMU}" type="wide"/>
  <p:notesSz cx="6858000" cy="9144000"/>
</p:presentation>"""


def presentation_rels(slide_count: int) -> str:
    rels = [
        f'  <Relationship Id="rId{i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i}.xml"/>'
        for i in range(1, slide_count + 1)
    ]
    rels.append(f'  <Relationship Id="rId{slide_count + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>')
    rels.append(f'  <Relationship Id="rId{slide_count + 2}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>')
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
""" + "\n".join(rels) + "\n</Relationships>"


def slide_xml(slide_no: int) -> str:
    name = escape(f"slide_{slide_no:02d}.png")
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld>
    <p:spTree>
      <p:nvGrpSpPr>
        <p:cNvPr id="1" name=""/>
        <p:cNvGrpSpPr/>
        <p:nvPr/>
      </p:nvGrpSpPr>
      <p:grpSpPr>
        <a:xfrm>
          <a:off x="0" y="0"/>
          <a:ext cx="0" cy="0"/>
          <a:chOff x="0" y="0"/>
          <a:chExt cx="0" cy="0"/>
        </a:xfrm>
      </p:grpSpPr>
      <p:pic>
        <p:nvPicPr>
          <p:cNvPr id="2" name="{name}"/>
          <p:cNvPicPr/>
          <p:nvPr/>
        </p:nvPicPr>
        <p:blipFill>
          <a:blip r:embed="rId1"/>
          <a:stretch><a:fillRect/></a:stretch>
        </p:blipFill>
        <p:spPr>
          <a:xfrm>
            <a:off x="0" y="0"/>
            <a:ext cx="{SLIDE_W_EMU}" cy="{SLIDE_H_EMU}"/>
          </a:xfrm>
          <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
        </p:spPr>
      </p:pic>
    </p:spTree>
  </p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sld>"""


def slide_rels(slide_no: int) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/slide_{slide_no:02d}.png"/>
</Relationships>"""


def minimal_theme() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Simple">
  <a:themeElements>
    <a:clrScheme name="Simple">
      <a:dk1><a:srgbClr val="111827"/></a:dk1>
      <a:lt1><a:srgbClr val="FFFFFF"/></a:lt1>
      <a:dk2><a:srgbClr val="374151"/></a:dk2>
      <a:lt2><a:srgbClr val="F7FBFF"/></a:lt2>
      <a:accent1><a:srgbClr val="2563EB"/></a:accent1>
      <a:accent2><a:srgbClr val="16A34A"/></a:accent2>
      <a:accent3><a:srgbClr val="F59E0B"/></a:accent3>
      <a:accent4><a:srgbClr val="DB2777"/></a:accent4>
      <a:accent5><a:srgbClr val="7C3AED"/></a:accent5>
      <a:accent6><a:srgbClr val="0891B2"/></a:accent6>
      <a:hlink><a:srgbClr val="2563EB"/></a:hlink>
      <a:folHlink><a:srgbClr val="7C3AED"/></a:folHlink>
    </a:clrScheme>
    <a:fontScheme name="Simple"><a:majorFont><a:latin typeface="Noto Sans CJK KR"/></a:majorFont><a:minorFont><a:latin typeface="Noto Sans CJK KR"/></a:minorFont></a:fontScheme>
    <a:fmtScheme name="Simple"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:fillStyleLst><a:lnStyleLst><a:ln w="6350"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst><a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst><a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme>
  </a:themeElements>
</a:theme>"""


def slide_master() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">
  <p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>
  <p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>
</p:sldMaster>"""


def slide_master_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>
</Relationships>"""


def slide_layout() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" preserve="1">
  <p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr></p:spTree></p:cSld>
  <p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr>
</p:sldLayout>"""


def slide_layout_rels() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>
</Relationships>"""


def build_pptx(slides: list[Path]) -> None:
    if PPTX_PATH.exists():
        PPTX_PATH.unlink()
    with zipfile.ZipFile(PPTX_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types(len(slides)))
        zf.writestr("_rels/.rels", xml_rels())
        zf.writestr("ppt/presentation.xml", presentation_xml(len(slides)))
        zf.writestr("ppt/_rels/presentation.xml.rels", presentation_rels(len(slides)))
        zf.writestr("ppt/theme/theme1.xml", minimal_theme())
        zf.writestr("ppt/slideMasters/slideMaster1.xml", slide_master())
        zf.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", slide_master_rels())
        zf.writestr("ppt/slideLayouts/slideLayout1.xml", slide_layout())
        zf.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", slide_layout_rels())
        for i, slide in enumerate(slides, start=1):
            zf.writestr(f"ppt/slides/slide{i}.xml", slide_xml(i))
            zf.writestr(f"ppt/slides/_rels/slide{i}.xml.rels", slide_rels(i))
            zf.write(slide, f"ppt/media/slide_{i:02d}.png")


def main() -> None:
    setup_font()
    slides = [
        slide_01(),
        slide_02(),
        slide_03(),
        slide_04(),
        slide_05(),
        slide_06(),
        slide_07(),
        slide_08(),
        slide_09(),
        slide_10(),
    ]
    build_pptx(slides)
    print(PPTX_PATH)


if __name__ == "__main__":
    main()
