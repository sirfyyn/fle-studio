"""The built-in model catalog (the FLE files on Hugging Face) and the fit estimate for this machine.

Every number below is taken from the model cards of sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF and
sirfyyn/Qwen3.8-Flash-Next-FLE-REAP448-GGUF (state 2026-10-10). Speed and memory figures were measured on one
machine only (RTX PRO 4500 Blackwell 32 GB, 120 GB DDR5, Ryzen 9 9950X3D, Linux, FLE engine); the UI says so
next to them. File sizes and SHA-256 are the Hub's own values.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .system import GiB

FLE_REPO = "sirfyyn/Qwen3.8-Flash-Next-FLE-GGUF"
REAP_REPO = "sirfyyn/Qwen3.8-Flash-Next-FLE-REAP448-GGUF"
REFERENCE_MACHINE = "RTX PRO 4500 Blackwell 32 GB, 120 GB DDR5, Ryzen 9 9950X3D, Linux"
REPORT_URL = "https://sirfyyn.github.io/FastLocalExperts/"


@dataclass(frozen=True)
class FileRef:
    repo: str
    path: str
    size: int
    sha256: str

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1]


@dataclass(frozen=True)
class Model:
    id: str
    name: str
    family: str
    repo: str
    files: tuple[FileRef, ...]
    summary: str
    kld: tuple[float, float, float] | None = None  # mean KLD vs BF16 on texts E / W / K (lower is better)
    speed: str = ""
    measured: dict[str, float] = field(default_factory=dict)  # on REFERENCE_MACHINE with the FLE engine
    mtp: bool = False  # can use the separate MTP head (FLE engine only)
    sampling: dict[str, float] = field(default_factory=dict)
    license: str = ""
    notes: tuple[str, ...] = ()

    @property
    def size(self) -> int:
        return sum(f.size for f in self.files)

    def public(self) -> dict[str, Any]:
        d = asdict(self)
        d["size"] = self.size
        d["files"] = [asdict(f) | {"name": f.name} for f in self.files]
        d["url"] = f"https://huggingface.co/{self.repo}"
        return d


QWEN_THINKING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5}
QWEN_LICENSE = "Qwen Community License 1.0"


def _qwen(id_: str, name: str, repo: str, files: tuple[FileRef, ...], **kw: Any) -> Model:
    return Model(id=id_, name=name, family="Qwen3.8-Flash-Next", repo=repo, files=files, mtp=True,
                 sampling=QWEN_THINKING, license=QWEN_LICENSE, **kw)


MTP_HEAD = FileRef(FLE_REPO, "mtp/mtp-Qwen3.8-Flash-Next-FLE-IQ4_NL.gguf", 1528277056,
                   "25217613e5e4659186db0b81ce2be77ab5955d01817732e4597b8f2bf0e292fa")
MTP_VOCAB = FileRef(FLE_REPO, "mtp/fle-mtp-vocab-32k.txt", 281159,
                    "8385f0457511fe600986bbd0ecb0b32766cbe31859c12e78a033685c28eef0d8")

CATALOG: tuple[Model, ...] = (
    _qwen(
        "fle-iq2_m", "FLE-IQ2_M", FLE_REPO,
        (
            FileRef(FLE_REPO, "IQ2_M/Qwen3.8-Flash-Next-FLE-IQ2_M-00001-of-00002.gguf", 42817327392,
                    "36fa6072301df9bbc54000164625cc4a7292030f84e834d8a2fe1be65d97cc9c"),
            FileRef(FLE_REPO, "IQ2_M/Qwen3.8-Flash-Next-FLE-IQ2_M-00002-of-00002.gguf", 28800138432,
                    "2fab43ec437fb7f0007686ac5ada3935ce7119fcfd8a3101931ab16485431155"),
        ),
        summary="Smallest and fastest FLE file. KL divergence clearly below the larger Unsloth UD-IQ1_M.",
        kld=(0.160, 0.163, 0.339),
        speed="+11.6 % decode against FLE-IQ3_M [95 %: +9.8 .. +13.4]",
        measured={"ram_gib": 38.7, "pinned_gib": 36.7, "gpu_mib": 31501, "load_s": 127},
    ),
    _qwen(
        "fle-iq3_xxs", "FLE-IQ3_XXS", FLE_REPO,
        (
            FileRef(FLE_REPO, "IQ3_XXS/Qwen3.8-Flash-Next-FLE-IQ3_XXS-00001-of-00002.gguf", 44931055808,
                    "363de45326e29a1d210077cc30340a25df2281514cf04969672b8d5d2abc505f"),
            FileRef(FLE_REPO, "IQ3_XXS/Qwen3.8-Flash-Next-FLE-IQ3_XXS-00002-of-00002.gguf", 34552744320,
                    "271b45f015bc1aa97fa44adf5c957330feb81ea85158b6428445fb441f8763b0"),
        ),
        summary="Same KL divergence as Unsloth UD-IQ3_XXS at 2.5 GB less; as fast as FLE-IQ3_M at 9.4 GB less.",
        kld=(0.090, 0.093, 0.227),
        speed="−0.7 % decode against FLE-IQ3_M [−2.1 .. +0.6], i.e. the same",
        measured={"ram_gib": 46.1, "pinned_gib": 43.9, "gpu_mib": 31279, "load_s": 241},
    ),
    _qwen(
        "fle-iq3_m-reap448", "FLE-IQ3_M-REAP448", REAP_REPO,
        (
            FileRef(REAP_REPO, "Qwen3.8-Flash-Next-FLE-IQ3_M-REAP448-00001-of-00002.gguf", 44914807136,
                    "b0d002c1c68896618ce3b481a811f3245d152a276cec947204c7f0590bed9516"),
            FileRef(REAP_REPO, "Qwen3.8-Flash-Next-FLE-IQ3_M-REAP448-00002-of-00002.gguf", 37087295680,
                    "27f52c8db7725da78c0b03bbc4e90ac5b4bb3cd1419f97e2d74e6e5203f5cbec"),
        ),
        summary="FLE-IQ3_M with 64 of 512 experts per layer pruned by REAP saliency. Lower KLD than UD-IQ3_XXS "
                "at the same size; answers slightly fewer knowledge questions than FLE-IQ3_M.",
        kld=(0.080, 0.086, 0.185),
        speed="+0.7 % decode against FLE-IQ3_M [−0.3 .. +1.8], i.e. the same",
        measured={"ram_gib": 48.0, "pinned_gib": 46.1, "gpu_mib": 31367, "load_s": 127},
        notes=("MKQA knowledge probe: −1.8 (German) / −3.2 (English) points against FLE-IQ3_M.",),
    ),
    _qwen(
        "fle-iq3_m", "FLE-IQ3_M", FLE_REPO,
        (
            FileRef(FLE_REPO, "IQ3_M/Qwen3.8-Flash-Next-FLE-IQ3_M-00001-of-00002.gguf", 44677859104,
                    "624718434053eb01798d20c0f35f4673db58ee629cd01f9aee236e80c9f474dc"),
            FileRef(FLE_REPO, "IQ3_M/Qwen3.8-Flash-Next-FLE-IQ3_M-00002-of-00002.gguf", 44194382624,
                    "7b3b094d2e5f0724720c217633677c450637cdd0212b02aad7a576510ecffa6a"),
        ),
        summary="Best quality of the FLE files: KL divergence within noise of Unsloth UD-IQ4_XS at 4.8 GB less.",
        kld=(0.051, 0.050, 0.129),
        speed="measured with the FLE engine and the MTP head on the reference machine: 91 tokens/s decode over six "
              "tasks, about 100 tokens/s on 24 greedy prompts",
        measured={"ram_gib": 54.3, "pinned_gib": 52.2, "gpu_mib": 31315, "load_s": 167},
    ),
)

BY_ID = {m.id: m for m in CATALOG}
BY_FILENAME = {f.name: m for m in CATALOG for f in m.files}


# ------------------------------------------------------------------------------------------------ fit


OS_RESERVE = 8 * GiB  # what the OS, the browser and FLE Studio itself keep
RUNTIME = 4 * GiB  # KV cache at 32k context, compute buffers, prompt cache floor


def fit(size: int, sysinfo: dict[str, Any]) -> dict[str, Any]:
    """Will a model of `size` bytes run on this machine? Returns a verdict and the numbers behind it.

    good: comfortable; tight: it should load but leaves little room (close other apps, smaller context);
    too_big: not enough memory. The estimate is deliberately simple and shown with its inputs.
    """
    need = size + RUNTIME
    ram = sysinfo["memory"]["total"]
    gpus = sysinfo.get("gpus") or []
    apple = next((g for g in gpus if g.get("unified")), None)
    if apple:
        limit = apple["metal_limit"]
        if need <= limit * 0.95:
            verdict = "good"
        elif need <= ram - OS_RESERVE:
            verdict = "tight"
        else:
            verdict = "too_big"
        detail = {"need": need, "gpu_limit": limit, "ram": ram}
        advice = ""
        if verdict == "tight":
            advice = (f"Above macOS's default GPU memory limit ({limit / GiB:.0f} GB). Raising it with "
                      f"`sudo sysctl iogpu.wired_limit_mb={int((ram - OS_RESERVE) / 2**20)}` lets the whole model "
                      "run on the GPU (resets at reboot).")
        return {"verdict": verdict, "kind": "unified", "detail": detail, "advice": advice}
    vram = max((g.get("memory_total", 0) for g in gpus), default=0)
    usable = max(0, ram - OS_RESERVE) + max(0, vram - 2 * GiB)
    if need <= usable * 0.9:
        verdict = "good"
    elif need <= usable:
        verdict = "tight"
    else:
        verdict = "too_big"
    advice = ""
    if verdict == "tight":
        advice = "Close other large programs and keep the context at 32k or less."
    elif verdict == "too_big":
        advice = (f"Needs about {need / GiB:.0f} GB of RAM and GPU memory together; "
                  f"this machine has {usable / GiB:.0f} GB.")
    return {"verdict": verdict, "kind": "split", "detail": {"need": need, "ram": ram, "vram": vram, "usable": usable},
            "advice": advice}


def catalog(sysinfo: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for m in CATALOG:
        d = m.public()
        d["fit"] = fit(m.size, sysinfo)
        out.append(d)
    return out


def extras() -> dict[str, dict[str, Any]]:
    return {"mtp_head": asdict(MTP_HEAD) | {"name": MTP_HEAD.name},
            "mtp_vocab": asdict(MTP_VOCAB) | {"name": MTP_VOCAB.name}}
