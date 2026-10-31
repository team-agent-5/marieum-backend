"""
임상 가이드를 chunk + embed + pgvector 적재.

입력 소스 (자동 감지, 혼용 가능):
    1. docs/guides/pdf/*.pdf          — PDF 원문 (페이지 단위, 500자 chunk)
    2. docs/guides/curated/*.jsonl    — 큐레이션 JSONL (한 줄 = 한 chunk)
    3. docs/guides/curated/*.json     — 큐레이션 JSON (리스트)

JSONL/JSON chunk 포맷:
    {"title": "...", "intent_tags": ["pain"], "content": "...",
     "source_url": "https://...", "page": 15}

둘 다 없으면 → SAMPLE_GUIDES (내장 데모 20건) 적재.

사용법:
    python scripts/ingest_guides.py              # 전체 적재 (기존 clinical_guide 는 유지)
    python scripts/ingest_guides.py --reset      # clinical_guide 전부 삭제 후 재적재
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sqlalchemy import delete

from app.db import AsyncSessionLocal
from app.models import Document
from app.services.embedding import get_embedder

try:
    from pypdf import PdfReader  # type: ignore[import-not-found]
except ImportError:
    PdfReader = None  # type: ignore[assignment]


# ───── 데모 가이드 (실제 공식 자료 투입 전 fallback) ─────
# 10 intent 전수 커버. 구음장애의 **원인 질환군**(뇌졸중·파킨슨병·치매·ALS·두경부암 등) 돌봄 관점.
# ⚠ 임상 검수 안 거친 데모용. 운영 전 docs/guides/pdf/ 또는 docs/guides/curated/ 에
# 전문가 검수 자료 투입 후 `--reset` 로 교체 바람.
SAMPLE_GUIDES: list[dict] = [
    # ─── water (삼킴장애 수분) — 뇌졸중/파킨슨/ALS 공통 ───
    {
        "title": "[데모] 뇌졸중 환자의 삼킴장애 수분 공급",
        "intent_tags": ["water"],
        "content": "뇌졸중 후유증 환자의 60~70%는 삼킴장애(dysphagia)가 있어 흡인성 폐렴 위험이 높다. 환자별 삼킴 능력 등급(normal·reduced·bedridden)에 맞는 점도의 음료를 제공하고, 상체를 30~45도 거상한 상태로 소량씩 공급한다. 사레들림이 반복되면 즉시 간호사에게 알린다. 와상(bedridden) 환자에게는 물을 공급하지 않는다.",
        "source_url": "demo://sample",
        "page": None,
    },
    {
        "title": "[데모] 파킨슨병 환자의 야간 수분 요청",
        "intent_tags": ["water"],
        "content": "파킨슨병 환자는 연하반사 지연과 야간 OFF 상태(약효 종료)로 사레들림 위험이 높다. 야간(22시~06시) 수분 요청은 환자 각성 상태를 먼저 확인하고, 반쯤 깨어있는 상태라면 침상에 상체를 거상한 뒤 소량(30~50ml) 제공한다. 반복 요청 시 담당 간호사와 상의하여 섭취량·시간을 기록한다.",
        "source_url": "demo://sample",
        "page": None,
    },
    # ─── toilet — 뇌졸중 편마비, 치매 환자 ───
    {
        "title": "[데모] 뇌졸중 편마비 환자 화장실 이동",
        "intent_tags": ["toilet"],
        "content": "편마비(hemiplegia) 환자의 배뇨·배변 호출 시 **낙상 위험** 평가가 최우선이다. 혼자 이동 가능 여부, 마비측 지지 필요성, 야간 조명 상태를 확인한다. 와상 환자는 체위 변경과 함께 간이 변기 사용을 지원한다. 변비 호소가 3일 이상 지속되면 간호사에 보고한다 (뇌졸중 환자 변비 흔함).",
        "source_url": "demo://sample",
        "page": None,
    },
    {
        "title": "[데모] 치매 환자 요실금 응대",
        "intent_tags": ["toilet"],
        "content": "중증 치매 환자의 요실금 호출은 신속하게 응대하되, 당황하지 않도록 안심시킨다. 환자가 자신의 상태를 인지하지 못할 수 있으므로 비난하지 않는다. 피부 손상 예방을 위해 즉시 기저귀/패드를 교체하고 회음부 세정을 수행한다. 야간 반복 시 수분 섭취 시간대 조정을 간호사에 보고한다.",
        "source_url": "demo://sample",
        "page": None,
    },
    # ─── pain — 치매·구음장애 환자의 비언어 통증 평가 ───
    {
        "title": "[데모] 통증 평가 NRS와 긴급 트리거",
        "intent_tags": ["pain"],
        "content": "환자가 통증을 호소하면 NRS(0~10) 척도로 강도를 평가한다. 부위·양상(둔통/박동성/찌르는)·지속시간을 함께 기록한다. NRS 7 이상이거나 흉통·두통+편측마비·호흡곤란이 동반되면 즉시 간호사·당직의에 보고 (뇌졸중 재발 가능성). 처방된 PRN 외 임의 투약은 금지한다.",
        "source_url": "demo://sample",
        "page": None,
    },
    {
        "title": "[데모] 중증 치매·구음장애 환자의 비언어 통증 평가",
        "intent_tags": ["pain", "anxiety"],
        "content": "말로 통증을 표현하기 어려운 환자(중증 치매·뇌졸중 후 실어·중증 구음장애)는 PAINAD 또는 FLACC 척도를 사용한다. 표정 찡그림, 신음, 몸의 긴장, 안절부절, 호흡 변화 등 **5개 비언어 지표**를 관찰한다. 평소 모습 대비 변화가 있으면 통증 가능성을 의심한다.",
        "source_url": "demo://sample",
        "page": None,
    },
    # ─── dyspnea — ALS·뇌졸중 후 흡인성 폐렴·심부전 ───
    {
        "title": "[데모] 호흡곤란 응급 대응",
        "intent_tags": ["dyspnea"],
        "content": "환자가 호흡곤란을 호소하면 즉시 상체를 30~45도 거상한다. SpO2 92% 미만이거나 호흡수가 분당 24회를 초과하면 즉시 당직의에게 알린다. 환자를 안심시키되 '괜찮다'며 응대를 지연하지 않는다. 뇌졸중 환자는 흡인성 폐렴, 치매 환자는 폐렴, ALS 환자는 호흡근 약화 가능성을 염두에 둔다.",
        "source_url": "demo://sample",
        "page": None,
    },
    {
        "title": "[데모] ALS·심부전 환자의 야간 호흡곤란 체위",
        "intent_tags": ["dyspnea"],
        "content": "ALS(루게릭) 환자는 야간 호흡근 약화로 숨가쁨이 자주 발생한다. 즉시 Fowler's position(반좌위)로 체위 변경하고, 베개 2~3개로 상체 지지. 환기를 확보한다. 발작성 야간 호흡곤란(PND)이 반복되면 심부전 가능성 — 반드시 간호사·당직의 보고. ALS 환자는 비침습적 환기(NIV) 적용 여부를 담당의와 사전 확인.",
        "source_url": "demo://sample",
        "page": None,
    },
    # ─── cold — 뇌졸중 자율신경, 치매 체온 조절 ───
    {
        "title": "[데모] 뇌졸중·노인 환자의 저체온증과 발열",
        "intent_tags": ["cold"],
        "content": "뇌졸중 후유증·고령 환자는 체온 조절 능력이 떨어져 **저체온(35.0도 미만)** 과 **발열(37.8도 이상)** 모두 위험하다. '추워요' 호소 시 체온 측정이 선행. 저체온은 담요·전기장판으로 보온, 발열은 과잉 보온을 피하고 간호사에 보고. 떨림(shivering)은 패혈증 전조일 수 있다 — 흡인성 폐렴 호발 환자군에서 특히 주의.",
        "source_url": "demo://sample",
        "page": None,
    },
    # ─── anxiety — 뇌졸중 후 섬망, 치매 야간 혼란 ───
    {
        "title": "[데모] 뇌졸중·치매 환자의 야간 섬망 응대",
        "intent_tags": ["anxiety"],
        "content": "급성기 뇌졸중 환자와 중증 치매 환자는 야간 섬망(delirium) 발생률이 높다. '무서워요', '누가 있어요' 등의 호소는 CAM(Confusion Assessment Method) 간이 평가 — 급성 발병·주의력 저하·사고 혼란·의식 변화. 조명을 적정 수준으로 밝히고 환자 이름을 부르며 현재 시간·장소를 안내한다. 신체 억제는 피하고 간호사·당직의와 상의.",
        "source_url": "demo://sample",
        "page": None,
    },
    {
        "title": "[데모] 구음장애 환자의 소통 좌절 안정화",
        "intent_tags": ["anxiety", "other"],
        "content": "뇌졸중 후 실어·구음장애, 파킨슨병 환자가 의사소통 좌절로 불안·공포를 보이면 서두르지 말고 '네/아니요' 질문으로 범위를 좁힌다. 환자의 눈을 바라보고 '천천히 다시 말씀해 주세요' 라고 말한다. 글자판·그림카드·AAC 앱 등 보조 수단을 즉시 제시한다.",
        "source_url": "demo://sample",
        "page": None,
    },
    # ─── fall — 파킨슨 보행장애, 뇌졸중 편마비 ───
    {
        "title": "[데모] 낙상 발생 시 초기 대응 (뇌졸중·파킨슨·치매)",
        "intent_tags": ["fall"],
        "content": "낙상 환자를 발견하면 **임의로 부축하거나 이동시키지 않는다**. 머리와 목이 움직이지 않도록 안내하고, 의식·출혈·활력징후를 먼저 확인한다. 뇌졸중 환자는 재발, 파킨슨 환자는 외상·골절, 치매 환자는 두부외상에 특히 유의. 의식저하·구토·심한 출혈이 있으면 당직의에 즉시 연락. 사고 시각·목격자·상황을 기록한다.",
        "source_url": "demo://sample",
        "page": None,
    },
    {
        "title": "[데모] 파킨슨·뇌졸중 환자의 Morse 낙상 위험 평가",
        "intent_tags": ["fall"],
        "content": "파킨슨병(보행 freezing·자세 불안정)과 뇌졸중 편마비 환자는 Morse Fall Scale 고위험군(총점 45점 이상)에 속한다. 침상 레일 유지, 야간 조명, 미끄럼 방지 매트, 신호 벨 접근성을 확보한다. 보행 보조기 필요한 환자는 반드시 보조 하에 이동. 혼자 침상 이탈 시 즉시 간호사 알림. 파킨슨 환자는 OFF 시간대(약효 종료) 특히 주의.",
        "source_url": "demo://sample",
        "page": None,
    },
    # ─── medication — 파킨슨 투약 타이밍, 노인 다약제 ───
    {
        "title": "[데모] 파킨슨병 환자의 레보도파 투약 타이밍",
        "intent_tags": ["medication"],
        "content": "파킨슨병 환자는 레보도파(levodopa)를 **정해진 시간에 정확히** 복용해야 ON 상태를 유지한다. 식사(특히 단백질)와 함께 복용하면 흡수가 저해된다. 환자가 '약 안 받았어요' 호소 시 복용 기록을 먼저 확인하고, 지연되면 즉시 간호사에 보고 — OFF 상태 지속 시 낙상·연하곤란 위험 증가. 임의 투약·시간 조정은 절대 금지.",
        "source_url": "demo://sample",
        "page": None,
    },
    {
        "title": "[데모] 노인 다약제 상호작용 — 뇌졸중·치매 환자",
        "intent_tags": ["medication"],
        "content": "뇌졸중·치매 환자는 보통 5종 이상 약제를 복용(항혈소판·혈압·당뇨·인지기능 등)하여 다약제(polypharmacy) 부작용 위험이 크다. '약 먹고 어지러워요' 유형 호출은 혈압·맥박·의식 상태를 즉시 확인하고 간호사·약사에 보고. 특히 와파린·아스피린 복용 환자의 출혈 징후(멍·검은변·코피)에 주의. 복용 시간 임의 조정 금지.",
        "source_url": "demo://sample",
        "page": None,
    },
    # ─── call (일반 호출) ───
    {
        "title": "[데모] 불명확한 호출 응대 매뉴얼",
        "intent_tags": ["call"],
        "content": "'잠깐 와주세요', '불러주세요' 등 의도가 명확하지 않은 호출은 환자 상태(의식·호흡·통증·낙상 위험)를 먼저 눈으로 확인한다. 긴급 신호(식은땀·창백·호흡 가쁨·의식 혼미)가 보이면 즉시 간호사 호출. 뇌졸중·치매 환자는 실어·실인증으로 **의도 표현이 더 어려울 수 있음**을 전제한다. 심리적 안정 요청이면 '괜찮아요, 바로 올게요'와 함께 짧은 대화로 안심시킨다.",
        "source_url": "demo://sample",
        "page": None,
    },
    # ─── other — 커뮤니케이션, 공통 ───
    {
        "title": "[데모] 구음장애 환자 비언어 신호 해석",
        "intent_tags": ["other"],
        "content": "뇌졸중·파킨슨·ALS 환자의 비언어 신호(눈 접촉·손짓·고개 움직임·호흡 패턴·근육 긴장)를 관찰한다. 발화가 불명확해도 **반복·강도·방향**이 메시지의 핵심이다. '예/아니요'를 눈 깜빡임 횟수로 미리 약속해두면 응대가 빨라진다. ALS 후기 환자는 안구 운동으로만 소통 가능할 수 있음을 염두에 둔다.",
        "source_url": "demo://sample",
        "page": None,
    },
    {
        "title": "[데모] 야간 간호 인력 응대 우선순위",
        "intent_tags": [],  # 모든 intent 공통
        "content": "야간(22시~06시) 인력 제한 환경에서 호출 우선순위: ① 호흡곤란·흉통·의식저하·낙상 → 즉시 간호사+당직의 ② 심한 통증(NRS≥7)·발열·출혈 → 간호사 즉시 ③ 수분·배뇨·체위 변경·불안 → 요양보호사 우선. 뇌졸중 환자의 '편측 쇠약 심해짐'은 재발 신호 — 긴급도 상향. 긴급도 상향은 보수적으로.",
        "source_url": "demo://sample",
        "page": None,
    },
    {
        "title": "[데모] AI 추천 가드레일 — 금지 사항",
        "intent_tags": [],
        "content": "AI 추천 카드는 **참고용**이며 돌봄인력이 최종 판단한다. 금지: ① 임의 투약·용량 변경(특히 파킨슨 레보도파·항응고제) ② 환자 억제 ③ 응급 상황에서 응대 지연 ④ 외부 반출 금지된 환자 정보 공유 ⑤ 명시적 처방 없는 산소/수액 공급. 가이드와 환자 상태가 상충되면 **환자 상태 우선**.",
        "source_url": "demo://sample",
        "page": None,
    },
]


# ───── Loaders ─────


def load_pdf(pdf_path: Path) -> list[dict]:
    """PDF → 페이지별 500자 chunk."""
    if PdfReader is None:
        raise RuntimeError("pypdf 미설치. `pip install -e .[rag-ingest]` 실행")
    reader = PdfReader(str(pdf_path))
    chunks: list[dict] = []
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if len(text) < 50:
            continue
        for j in range(0, len(text), 500):
            chunks.append(
                {
                    "title": pdf_path.stem,
                    "intent_tags": [],
                    "content": text[j : j + 500],
                    "source_url": f"file://{pdf_path.name}",
                    "page": i,
                }
            )
    return chunks


def load_jsonl(path: Path) -> list[dict]:
    """JSONL — 한 줄에 한 chunk."""
    chunks: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno} JSON 파싱 실패: {e}") from e
            _validate_chunk(obj, f"{path}:{lineno}")
            chunks.append(obj)
    return chunks


def load_json(path: Path) -> list[dict]:
    """JSON — 리스트 전체."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"{path}: 최상위는 리스트여야 함")
    for i, obj in enumerate(data):
        _validate_chunk(obj, f"{path}[{i}]")
    return data


def _validate_chunk(obj: dict, loc: str) -> None:
    if "content" not in obj or not isinstance(obj["content"], str):
        raise ValueError(f"{loc}: 'content' 필드(str) 필수")
    obj.setdefault("title", None)
    obj.setdefault("intent_tags", [])
    obj.setdefault("source_url", None)
    obj.setdefault("page", None)


# ───── Main ─────


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset", action="store_true", help="기존 clinical_guide 전부 삭제 후 재적재"
    )
    args = parser.parse_args()

    emb = get_embedder()
    guides_dir = Path(__file__).resolve().parent.parent / "docs" / "guides"
    pdf_dir = guides_dir / "pdf"
    curated_dir = guides_dir / "curated"

    all_chunks: list[dict] = []

    # 1. PDF (docs/guides/pdf/*.pdf)
    pdf_files = sorted(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
    for pdf in pdf_files:
        n_before = len(all_chunks)
        all_chunks.extend(load_pdf(pdf))
        print(f"  PDF    {pdf.name:40s} → {len(all_chunks) - n_before:>3} chunks")

    # 2. Curated JSONL/JSON (docs/guides/curated/*)
    curated_files = []
    if curated_dir.exists():
        curated_files = sorted(curated_dir.glob("*.jsonl")) + sorted(curated_dir.glob("*.json"))
    for cf in curated_files:
        n_before = len(all_chunks)
        if cf.suffix == ".jsonl":
            all_chunks.extend(load_jsonl(cf))
        else:
            all_chunks.extend(load_json(cf))
        print(f"  JSON   {cf.name:40s} → {len(all_chunks) - n_before:>3} chunks")

    # 3. Fallback — 입력 없으면 데모 샘플
    if not all_chunks:
        print("⚠ docs/guides/pdf/ 와 docs/guides/curated/ 둘 다 비어있음")
        print("  → 내장 데모 샘플 20건으로 적재 (임상 검수 안 됨, 운영 전 교체 필수)")
        all_chunks = SAMPLE_GUIDES

    # ───── DB 적재 ─────
    async with AsyncSessionLocal() as session:
        if args.reset:
            print("⚠ --reset: 기존 clinical_guide 전부 삭제")
            await session.execute(delete(Document).where(Document.source == "clinical_guide"))
            await session.flush()

        for ch in all_chunks:
            v = emb.encode(ch["content"], normalize_embeddings=True).tolist()
            session.add(
                Document(
                    source="clinical_guide",
                    patient_id=None,
                    intent_tags=ch.get("intent_tags", []),
                    title=ch.get("title"),
                    content=ch["content"],
                    source_url=ch.get("source_url"),
                    page=ch.get("page"),
                    embedding=v,
                )
            )
        await session.commit()
        print(f"✅ clinical_guide {len(all_chunks)}건 적재 완료")


if __name__ == "__main__":
    asyncio.run(main())
