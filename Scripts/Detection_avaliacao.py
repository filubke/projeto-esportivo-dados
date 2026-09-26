# -*- coding: utf-8 -*-

import sys
import io
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import unicodedata

import csv
import html as html_lib
import json
import os
import time
from collections import Counter, defaultdict, deque
from time import perf_counter

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from ultralytics import RTDETR



# ============================================================
# CONFIGURAÇÃO GERAL
# ============================================================
VIDEO_PATH = r"E:\Projeto\Videos\Jogo_Aranha-05-09-26.mp4"
OUTPUT_DIR = r"E:\Projeto\Videos\Saidas"
CSV_DIR    = r"E:\Projeto\CSV"
HTML_DIR   = r"E:\Projeto\HTML"
JSON_DIR   = r"E:\Projeto\JSON"

DEVICE = "cpu" 
PULAR_FRAMES     = 1
FRAME_REFERENCIA = 0
# Ajustados pelo notebook quando houver GPU disponível.
HALF_INFERENCIA = False
# Use um número para validar um clipe curto; None processa até o fim.
MAX_FRAMES_PROCESSAMENTO = 500

CONFIANCA_JOGADOR = 0.25
CONFIANCA_BOLA    = 0.12
IOU_DETECCAO      = 0.40
NMS_JOGADORES     = 0.50
NMS_BOLA          = 0.30


INTERVALO_APARENCIA = 3
MAX_HIST_TRACK       = 20
MINIMO_VOTOS_TIME    = 3
MARGEM_MINIMA_TIME   = 0.06
PERMITIR_TROCA_TIME  = True
MATCH_SCORE_MIN      = 0.18
MATCH_IOU_MIN        = 0.05
MATCH_DISTANCIA_MAX  = 0.22
TRACK_BUFFER         = 60

# Pesos da associação entre uma detecção/trilha e uma referência.
PESO_ASSOC_IOU        = 0.30
PESO_ASSOC_DIST       = 0.20
PESO_ASSOC_VEL        = 0.15
PESO_ASSOC_APARENCIA  = 0.25
PESO_ASSOC_PAPEL      = 0.10
NORFAIR_DISTANCE_THRESHOLD = 80.0
BALL_MAX_PREDICT_FRAMES = 8
BALL_MAX_JUMP_NORMALIZADO = 0.12

SALVAR_CSV_DETECCOES = True
SALVAR_CSV_RESUMO    = True
SALVAR_JSON_METRICAS = True
SALVAR_VIDEO = True
# Preview ao vivo: "local" usa cv2.imshow; "colab" atualiza a célula inline.
EXIBIR_PREVIEW = False
PREVIEW_MODO = "local"
PREVIEW_INTERVALO = 1

MODELO_YOLO = r"E:\Projeto\Modelos\yolo26s.pt"
MODELO_RTDETR_BOLA = r"E:\Projeto\Modelos\rtdetr-l.pt"
CLASSE_BOLA_RTDETR = 32
_CACHE_RTDETR_BOLA = {}
RTDTER_BOLA = False

# Modelo _best
# 0: ball
#1: goalkeeper
#2: player
#3: referee


# ============================================================
# PAPÉIS (ROLES)
# ============================================================
ROLES = [
    {"id": "time0",    "label": "TIME 1 — jogadores",         "tipo": "time",    "obrigatorio": True},
    {"id": "time1",    "label": "TIME 2 — jogadores",         "tipo": "time",    "obrigatorio": True},
    {"id": "goleiro0", "label": "GOLEIRO — Time 1",           "tipo": "goleiro", "vinculado_a": "time0", "obrigatorio": False},
    {"id": "goleiro1", "label": "GOLEIRO — Time 2",           "tipo": "goleiro", "vinculado_a": "time1", "obrigatorio": False},
    {"id": "arbitro",  "label": "ÁRBITRO",                    "tipo": "neutro",  "obrigatorio": False},
    {"id": "excluir",  "label": "EXCLUIR (gandula/comissão)", "tipo": "excluir", "obrigatorio": False},
]


def rotulo_grupo(role_id: str | None) -> tuple[str, str, bool]:
    """Retorna (rótulo de exibição, grupo p/ métricas, excluir_das_metricas?)."""
    if role_id is None:
        return "?", "indefinido", False
    role = next((r for r in ROLES if r["id"] == role_id), None)
    if role is None:
        return "?", "indefinido", False
    if role["tipo"] == "excluir":
        return "EXCLUÍDO", "excluido", True
    if role["tipo"] == "neutro":
        return "ÁRBITRO", "arbitro", False
    if role["tipo"] == "goleiro":
        return f"GOL-{role['vinculado_a'].upper()}", f"goleiro_{role['vinculado_a']}", False
    return role_id.upper(), role_id, False


# ============================================================
# CONFIGURAÇÃO DOS TESTES
# Cada teste é uma combinação isolada de detector x tracker x aparência.
# Descomente as opções conforme for testando — todas têm hook pronto no código.
# ============================================================
TESTES = [
    # ------ Selecionados
    # Classe _best -> {"player": 2, "ball": 0, "goalkeeper": 1, "referee": 3}
    # classe geral -> {"player": 0, "ball": 32, "goalkeeper": None, "referee": set()}
    {
        "nome_teste": "YOLO26S_BoTSORT_HSV_1280",
        "detector":   "yolo",
        "model_path": MODELO_YOLO,
        "rtdetr_bola_model_path": MODELO_RTDETR_BOLA,
        "imgsz":      1280,
        "tracker":    "botsort",
        "aparencia":  "hsv",
        "usar_rtdetr_bola": RTDTER_BOLA,
        "classes": {"player": 0, "ball": 32, "goalkeeper": None, "referee": set()},
    },
    # {
    #     "nome_teste": "YOLO26M_NorFair_HSV_1280", 
    #     "detector": "yolo",
    #     "model_path": MODELO_YOLO,
    #     "rtdetr_bola_model_path": MODELO_RTDETR_BOLA, 
    #     "imgsz": 1280,
    #     "tracker": "norfair", 
    #     "aparencia": "hsv",
    #     "usar_rtdetr_bola": RTDTER_BOLA,
    #     "classes": {"player": 0, "ball": 32, "goalkeeper": None, "referee": set()},
    # },
    # {
    #     "nome_teste": "YOLO26M_NorFair_Siames_1280", 
    #     "detector": "yolo",
    #     "model_path": MODELO_YOLO,
    #     "rtdetr_bola_model_path": MODELO_RTDETR_BOLA, 
    #     "imgsz": 1280,
    #     "tracker": "norfair", 
    #     "aparencia": "siames",
    #     "usar_rtdetr_bola": RTDTER_BOLA,
    #     "classes": {"player": 0, "ball": 32, "goalkeeper": None, "referee": set()},
    # },
    
]

cv2.setNumThreads(max(1, min(6, os.cpu_count() or 2)))


# ============================================================
# UTILITÁRIOS GEOMÉTRICOS
# ============================================================
def criar_pastas():
    for pasta in (OUTPUT_DIR, CSV_DIR, HTML_DIR, JSON_DIR):
        os.makedirs(pasta, exist_ok=True)

def criar_anotadores():
    palette_hex = ["#2E91CA", "#895129", "#00FF00", "#FFFF00", "#FF00FF", "#888888"]
    palette = sv.ColorPalette.from_hex(palette_hex)
    ellipse_annotator = sv.EllipseAnnotator(color=palette, thickness=2)
    label_annotator = sv.LabelAnnotator(
        color=palette, text_scale=0.4,
        text_color=sv.Color.BLACK,
        text_position=sv.Position.BOTTOM_CENTER
    )
    triangle_annotator = sv.TriangleAnnotator(
        color=sv.Color.from_hex("#FFD700"), base=14, height=12
    )
    return ellipse_annotator, label_annotator, triangle_annotator


def cor_index_por_papel(role_id: str) -> int:
    """Mapeia role_id para um índice de cor fixo, para o vídeo ficar consistente entre frames."""
    ordem = ["time0", "time1", "goleiro0", "goleiro1", "arbitro", "excluir"]
    return ordem.index(role_id) if role_id in ordem else 0

def calcular_imgsz(largura, altura, limite=1280):
    return max(32, min((max(largura, altura) // 32) * 32, limite))


def centro_box(box):
    x1, y1, x2, y2 = map(float, box)
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)


def area_box(box):
    x1, y1, x2, y2 = map(float, box)
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def iou_boxes(a, b):
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = area_box(a) + area_box(b) - inter
    return inter / union if union > 0 else 0.0


def distancia_normalizada(a, b, largura, altura):
    return float(np.linalg.norm(centro_box(a) - centro_box(b)) / max(1.0, np.hypot(largura, altura)))


def limitar_box(box, largura, altura):
    x1, y1, x2, y2 = map(int, box)
    return (max(0, min(x1, largura - 1)), max(0, min(y1, altura - 1)),
            max(0, min(x2, largura)), max(0, min(y2, altura)))



def remover_acentos_cv2(texto: str) -> str:
    """
    cv2.putText (fontes Hershey) não suporta caracteres acentuados/UTF-8 —
    renderiza como '??'. Esta função converte para ASCII puro só para exibição
    na janela; o terminal (print) continua acentuado normalmente.
    """
    nfkd = unicodedata.normalize("NFKD", texto)
    sem_acentos = "".join(c for c in nfkd if not unicodedata.combining(c))
    return sem_acentos.replace("—", "-")  # travessão também não renderiza

def imprimir_progresso(nome, frame, total, processados, inicio, extra=""):
    elapsed = time.time() - inicio
    vel = processados / elapsed if elapsed > 0 else 0.0
    eta = int(max(0, total - frame) / vel) if vel > 0 else 0
    print(f"[{nome}] {100*frame/max(1,total):6.2f}% | frame {frame}/{total} | "
          f"{vel:.2f} FPS | ETA {eta//60}m{eta%60:02d}s | {extra}", flush=True)



# ============================================================
# CROP E APARÊNCIA — HSV (baseline)
# ============================================================
def crop_torso(frame, box):
    altura, largura = frame.shape[:2]
    x1, y1, x2, y2 = limitar_box(box, largura, altura)

    if x2 <= x1 or y2 <= y1:
        return None

    h, w = y2 - y1, x2 - x1

    if h < 24 or w < 16:
        return None

    topo = y1 + int(0.16 * h)
    base = y1 + int(0.68 * h)
    margem = int(0.08 * w)

    left = min(x2 - 1, x1 + margem)
    right = max(x1 + 1, x2 - margem)

    crop = frame[topo:base, left:right]

    if crop.size == 0:
        return None

    return crop



def assinatura_hsv(crop):
    if crop is None or crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    grama = cv2.inRange(hsv, np.array([30, 35, 35]), np.array([90, 255, 255]))
    escuro = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 45]))
    mascara = cv2.bitwise_not(cv2.bitwise_or(grama, escuro))
    if cv2.countNonZero(mascara) < 15:
        return None
    vetor = np.concatenate([
        cv2.normalize(cv2.calcHist([hsv], [0], mascara, [18], [0, 180]), None).flatten(),
        cv2.normalize(cv2.calcHist([hsv], [1], mascara, [8], [0, 256]), None).flatten(),
        cv2.normalize(cv2.calcHist([hsv], [2], mascara, [8], [0, 256]), None).flatten(),
    ]).astype(np.float32)
    return vetor / (np.linalg.norm(vetor) + 1e-8)


def similaridade(a, b):
    if a is None or b is None:
        return -1.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


# ============================================================
# APARÊNCIA NEURAL — SigLIP / DINOv2 / MobileNet / Redes Siamesas
# ============================================================
class ExtratorSigLIP_CPU:
    """Embedding visual do SigLIP."""

    def __init__(self):
        import torch
        from PIL import Image
        from transformers import (
            SiglipVisionModel,
            AutoImageProcessor,
        )

        self.torch = torch
        self.Image = Image

        nome_modelo = "google/siglip-base-patch16-224"

        print("[siglip] carregando em CPU...", flush=True)

        self.processor = AutoImageProcessor.from_pretrained(
            nome_modelo
        )

        self.model = SiglipVisionModel.from_pretrained(
            nome_modelo
        ).to("cpu").eval()

        self.dim = getattr(
            self.model.config.vision_config,
            "hidden_size",
            768,
        )

    def extrair(self, crops):
        if not crops:
            return np.empty(
                (0, self.dim),
                dtype=np.float32,
            )

        imgs = [
            self.Image.fromarray(
                cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
            )
            for c in crops
            if c is not None and c.size > 0
        ]

        if not imgs:
            return np.empty(
                (0, self.dim),
                dtype=np.float32,
            )

        with self.torch.inference_mode():
            entradas = self.processor(
                images=imgs,
                return_tensors="pt",
            )

            saida = self.model(**entradas)

            embeddings = self.torch.nn.functional.normalize(
                saida.pooler_output.float(),
                p=2,
                dim=1,
            )

            return embeddings.cpu().numpy().astype(np.float32)



class ExtratorDINOv2_CPU:
    """384D (dinov2-small). Alternativa ao SigLIP, captura textura/padrão."""
    def __init__(self, modelo="facebook/dinov2-small"):
        import torch
        from PIL import Image
        from transformers import AutoModel, AutoImageProcessor
        self.torch, self.Image = torch, Image
        print(f"[dinov2] carregando {modelo} em CPU...", flush=True)
        self.processor = AutoImageProcessor.from_pretrained(modelo)
        self.model = AutoModel.from_pretrained(modelo).to("cpu").eval()

    def extrair(self, crops):
        if not crops:
            return np.empty((0, 384), dtype=np.float32)
        imgs = [self.Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops]
        with self.torch.inference_mode():
            entradas = self.processor(images=imgs, return_tensors="pt")
            saida = self.model(**entradas)
            cls = saida.last_hidden_state[:, 0, :]
            t = self.torch.nn.functional.normalize(cls.float(), p=2, dim=1)
            return t.cpu().numpy().astype(np.float32)


class ExtratorMobileNet_CPU:
    """576D (MobileNetV3-Small, ImageNet). Meio-termo entre HSV e SigLIP/DINOv2."""
    def __init__(self):
        import torch
        import torchvision.models as models
        import torchvision.transforms as T
        from PIL import Image
        self.torch, self.Image = torch, Image
        print("[mobilenet] carregando MobileNetV3-Small...", flush=True)
        modelo = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        modelo.classifier = torch.nn.Identity()
        self.model = modelo.to("cpu").eval()
        self.transform = T.Compose([
            T.Resize((128, 128)), T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extrair(self, crops):
        if not crops:
            return np.empty((0, 576), dtype=np.float32)
        imgs = [self.Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops]
        tensores = self.torch.stack([self.transform(im) for im in imgs])
        with self.torch.inference_mode():
            saida = self.model(tensores)
            t = self.torch.nn.functional.normalize(saida.float(), p=2, dim=1)
            return t.cpu().numpy().astype(np.float32)

class ExtratorSiames_CPU:
    """Extrai embeddings (geralmente 512D) usando uma Rede Siamesa focada em Person Re-ID."""
    def __init__(self, weights_path=r"E:\Projeto\Modelos\osnet_x0_25_msmt17.pt"):
        import torch
        import torchvision.transforms as T
        from PIL import Image
        
        # Exemplo usando a biblioteca torchreid (ou similar que suporte seu peso)
        # pip install torchreid
        import torchreid
        
        self.torch, self.Image = torch, Image
        print("[siames] carregando modelo OSNet para Re-ID...", flush=True)
        
        # Carrega a arquitetura siamesa
        self.model = torchreid.models.build_model(
            name="osnet_x0_25", num_classes=1000, loss="softmax", pretrained=False
        )
        torchreid.utils.load_pretrained_weights(self.model, weights_path)
        self.model.eval().to("cpu")
        
        self.transform = T.Compose([
            T.Resize((256, 128)), # Redes Re-ID geralmente usam proporção 2:1
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extrair(self, crops):
        if not crops:
            return np.empty((0, 512), dtype=np.float32) # Dimensão varia pelo modelo
            
        imgs = [self.Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops]
        tensores = self.torch.stack([self.transform(im) for im in imgs])
        
        with self.torch.inference_mode():
            # A rede siamesa cospe o vetor descritor diretamente
            saida = self.model(tensores)
            t = self.torch.nn.functional.normalize(saida.float(), p=2, dim=1)
            return t.cpu().numpy().astype(np.float32)

class ExtratorLBP_CPU:
    """Textura via Local Binary Patterns."""

    def __init__(self, radius=1, n_points=8, method="uniform"):
        from skimage.feature import local_binary_pattern

        self.lbp_fn = local_binary_pattern
        self.radius = radius
        self.n_points = n_points
        self.method = method

        # Para method="uniform", o número de padrões é P + 2
        self.num_bins = n_points + 2

    def extrair(self, crops):
        if not crops:
            return np.empty(
                (0, self.num_bins),
                dtype=np.float32,
            )

        feats = []

        for c in crops:
            if c is None or c.size == 0:
                continue

            gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)

            # Redimensiona para estabilizar a escala da textura
            gray = cv2.resize(
                gray,
                (64, 64),
                interpolation=cv2.INTER_AREA,
            )

            lbp = self.lbp_fn(
                gray,
                P=self.n_points,
                R=self.radius,
                method=self.method,
            )

            if self.method == "uniform":
                hist, _ = np.histogram(
                    lbp.ravel(),
                    bins=np.arange(0, self.num_bins + 1),
                    range=(0, self.num_bins),
                )
            else:
                # Para method="default", existem 2^P padrões
                num_bins = 2 ** self.n_points
                hist, _ = np.histogram(
                    lbp.ravel(),
                    bins=np.arange(0, num_bins + 1),
                    range=(0, num_bins),
                )

            hist = hist.astype(np.float32)
            hist /= hist.sum() + 1e-8
            feats.append(hist)

        if not feats:
            return np.empty(
                (0, self.num_bins),
                dtype=np.float32,
            )

        return np.stack(feats, axis=0)


class ExtratorHOG_CPU:
    """Forma e estrutura via HOG."""

    def __init__(
        self,
        cell_size=(8, 8),
        block_size=(2, 2),
        orientations=9,
        resize_shape=(64, 64),
    ):
        from skimage.feature import hog

        self.hog_fn = hog
        self.cell_size = cell_size
        self.block_size = block_size
        self.orientations = orientations
        self.resize_shape = resize_shape

    def extrair(self, crops):
        if not crops:
            return np.empty((0, 0), dtype=np.float32)

        feats = []

        for c in crops:
            if c is None or c.size == 0:
                continue

            gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)

            # resize_shape é (largura, altura) para cv2.resize
            gray = cv2.resize(
                gray,
                self.resize_shape,
                interpolation=cv2.INTER_AREA,
            )

            h = self.hog_fn(
                gray,
                orientations=self.orientations,
                pixels_per_cell=self.cell_size,
                cells_per_block=self.block_size,
                visualize=False,
                feature_vector=True,
            )

            h = h.astype(np.float32)
            h /= np.linalg.norm(h) + 1e-8
            feats.append(h)

        if not feats:
            return np.empty((0, 0), dtype=np.float32)

        return np.stack(feats, axis=0)


_EXTRATORES_NEURAIS = {
    "embeddings": ExtratorSigLIP_CPU,
    "dinov2":     ExtratorDINOv2_CPU,
    "mobilenet":  ExtratorMobileNet_CPU,
    "siames":     ExtratorSiames_CPU,
    "lbp":        ExtratorLBP_CPU,
    "hog":        ExtratorHOG_CPU,
}


def criar_extrator(opcao_aparencia: str):
    classe = _EXTRATORES_NEURAIS.get(opcao_aparencia)
    return classe() if classe else None


def assinatura_por_opcao(crop, opcao, extrator):
    if opcao in ("hsv", "kmeans_hsv"):
        return assinatura_hsv(crop)
    if opcao in _EXTRATORES_NEURAIS and extrator is not None and crop is not None:
        valores = extrator.extrair([crop])
        return valores[0] if len(valores) else None
    return None


# ============================================================
# CLASSIFICADOR DE PAPÉIS — generaliza Time/Goleiro/Árbitro/Excluir
# ============================================================
class ClassificadorPapeis:
    def __init__(self):
        self.ancoras: dict[str, np.ndarray] = {}
        self.assinaturas = defaultdict(lambda: deque(maxlen=MAX_HIST_TRACK))
        self.votos = defaultdict(lambda: deque(maxlen=MAX_HIST_TRACK))
        self.confirmados: dict[int, str] = {}

    def configurar_ancoras(self, centroides: dict):
        self.ancoras = {k: v for k, v in centroides.items() if v is not None}

    def adicionar(self, tid, assinatura):
        if assinatura is not None:
            self.assinaturas[int(tid)].append(assinatura)

    def scores_por_papel(self, tid):
        dados = self.assinaturas.get(int(tid))
        if not dados:
            return {}
        medio = np.mean(np.asarray(dados), axis=0)
        medio /= np.linalg.norm(medio) + 1e-8
        return {rid: similaridade(medio, ancora) for rid, ancora in self.ancoras.items()}

    def prever(self, tid):
        scores_dict = self.scores_por_papel(tid)
        if len(scores_dict) < 2:
            return None, 0.0
        scores = sorted(((score, rid) for rid, score in scores_dict.items()), reverse=True)
        margem = scores[0][0] - scores[1][0]
        return (scores[0][1], float(margem)) if margem >= MARGEM_MINIMA_TIME else (None, margem)

    def atualizar(self, tid, pred):
        tid = int(tid)
        atual = self.confirmados.get(tid)
        if pred is None:
            return atual
        self.votos[tid].append(pred)
        vencedor, qtd = Counter(self.votos[tid]).most_common(1)[0]
        if atual is None:
            if qtd >= MINIMO_VOTOS_TIME:
                self.confirmados[tid] = vencedor
                return vencedor
            return None
        if pred == atual:
            return atual
        if PERMITIR_TROCA_TIME and qtd >= MINIMO_VOTOS_TIME:
            self.confirmados[tid] = vencedor
        return self.confirmados.get(tid, atual)


class ClassificadorKMeansDuploPapeis:
    """Sub-clustering (K=2) do HSV por papel — reduz efeito de iluminação
    inconsistente comparando contra o melhor sub-centroide, não a média única."""
    def __init__(self):
        self.subancoras_por_papel: dict[str, list] = {}
        self.assinaturas = defaultdict(lambda: deque(maxlen=MAX_HIST_TRACK))
        self.votos = defaultdict(lambda: deque(maxlen=MAX_HIST_TRACK))
        self.confirmados: dict[int, str] = {}

    def configurar_ancoras(self, banco_por_papel: dict):
        try:
            from sklearn.cluster import KMeans
        except ImportError as exc:
            raise RuntimeError("pip install scikit-learn") from exc
        self.subancoras_por_papel = {}
        for role_id, vetores in banco_por_papel.items():
            validos = [v for v in vetores if v is not None]
            if len(validos) < 2:
                self.subancoras_por_papel[role_id] = validos
                continue
            k = min(2, len(validos))
            km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(np.asarray(validos))
            self.subancoras_por_papel[role_id] = list(km.cluster_centers_)

    def adicionar(self, tid, assinatura):
        if assinatura is not None:
            self.assinaturas[int(tid)].append(assinatura)

    def scores_por_papel(self, tid):
        dados = self.assinaturas.get(int(tid))
        if not dados:
            return {}
        medio = np.mean(np.asarray(dados), axis=0)
        medio /= np.linalg.norm(medio) + 1e-8
        return {
            role_id: max((similaridade(medio, c) for c in subcentroides), default=-1.0)
            for role_id, subcentroides in self.subancoras_por_papel.items()
            if subcentroides
        }

    def prever(self, tid):
        scores_dict = self.scores_por_papel(tid)
        if len(scores_dict) < 2:
            return None, 0.0
        scores = sorted(((score, rid) for rid, score in scores_dict.items()), reverse=True)
        margem = scores[0][0] - scores[1][0]
        return (scores[0][1], float(margem)) if margem >= MARGEM_MINIMA_TIME else (None, margem)

    def atualizar(self, tid, pred):
        tid = int(tid)
        atual = self.confirmados.get(tid)
        if pred is None:
            return atual
        self.votos[tid].append(pred)
        vencedor, qtd = Counter(self.votos[tid]).most_common(1)[0]
        if atual is None:
            if qtd >= MINIMO_VOTOS_TIME:
                self.confirmados[tid] = vencedor
                return vencedor
            return None
        if pred == atual:
            return atual
        if PERMITIR_TROCA_TIME and qtd >= MINIMO_VOTOS_TIME:
            self.confirmados[tid] = vencedor
        return self.confirmados.get(tid, atual)


def criar_classificador(opcao_aparencia: str):
    return ClassificadorKMeansDuploPapeis() if opcao_aparencia == "kmeans_hsv" else ClassificadorPapeis()


# ============================================================
# TRACKERS
# ============================================================
def _criar_bytetrack(fps):
    from trackers import ByteTrackTracker
    for kwargs in [
        dict(track_activation_threshold=.20, lost_track_buffer=TRACK_BUFFER,
             minimum_matching_threshold=.75, frame_rate=max(1, int(fps)),
             minimum_consecutive_frames=2),
        dict(track_activation_threshold=.20, lost_track_buffer=TRACK_BUFFER,
             minimum_matching_threshold=.75),
        {},
    ]:
        try:
            return ("supervision", ByteTrackTracker(**kwargs))
        except TypeError:
            pass
    return ("supervision", ByteTrackTracker())

def detectar_bola_rtdetr(modelo, frame, imgsz, classe_ball=0):
    resultados = modelo.predict(
        source=frame,
        conf=CONFIANCA_BOLA,
        imgsz=imgsz,
        device=DEVICE,
        half=HALF_INFERENCIA,
        verbose=False,
    )[0]

    bola = sv.Detections.from_ultralytics(resultados)

    if len(bola) == 0:
        return bola

    if bola.class_id is not None:
        bola = bola[bola.class_id == int(classe_ball)]

    return bola


class NorfairAdapter:
    """Adapta centros das caixas para o tracker Norfair e reconstrói caixas."""
    def __init__(self, distance_threshold=NORFAIR_DISTANCE_THRESHOLD):
        try:
            from norfair import Tracker
        except ImportError as exc:
            raise RuntimeError("Norfair requer: pip install norfair") from exc
        self.tracker = Tracker(
            distance_function="euclidean",
            distance_threshold=distance_threshold,
            hit_counter_max=TRACK_BUFFER,
            initialization_delay=1,
        )

    def update(self, jogadores):
        from norfair import Detection
        detections = []
        for box, conf in zip(jogadores.xyxy, jogadores.confidence):
            x1, y1, x2, y2 = map(float, box)
            centro = np.array([[(x1 + x2) / 2, (y1 + y2) / 2]], dtype=np.float32)
            detections.append(Detection(
                points=centro,
                scores=np.array([float(conf)], dtype=np.float32),
                data={"box": np.array([x1, y1, x2, y2], dtype=np.float32)},
            ))
        objetos = self.tracker.update(detections=detections)
        boxes, tids, confs = [], [], []
        for obj in objetos:
            if obj.last_detection is None:
                continue
            data = obj.last_detection.data or {}
            box = data.get("box")
            if box is None:
                cx, cy = obj.estimate[0]
                box = np.array([cx - 20, cy - 50, cx + 20, cy + 50], dtype=np.float32)
            boxes.append(box)
            tids.append(int(obj.id))
            confs.append(float(obj.last_detection.scores[0]))
        if not boxes:
            return sv.Detections.empty()
        return sv.Detections(
            xyxy=np.asarray(boxes, dtype=np.float32),
            tracker_id=np.asarray(tids, dtype=int),
            confidence=np.asarray(confs, dtype=np.float32),
            class_id=np.zeros(len(boxes), dtype=int),
        )

def _criar_botsort():
    """BoT-SORT é nativo da Ultralytics — chamado via model.track(), não update()."""
    return ("ultralytics_native", "botsort.yaml")

def _criar_ocsort():
    try:
        from boxmot import OcSort
        from pathlib import Path
    except ImportError as exc:
        raise RuntimeError("OC-SORT (via boxmot) requer: pip install boxmot") from exc
    tracker = OcSort(
        det_thresh=CONFIANCA_JOGADOR,
        iou_threshold=0.3,
        max_age=30,
        min_hits=3,
    )
    return ("ocsort_boxmot", tracker)

def _criar_deepsort():
    try:
        from deep_sort_realtime.deepsort_tracker import DeepSort
    except ImportError as exc:
        raise RuntimeError("DeepSORT requer: pip install deep-sort-realtime") from exc
    return ("deepsort", DeepSort(max_age=TRACK_BUFFER))


def _criar_strongsort():
    try:
        from boxmot import StrongSort
    except ImportError as exc:
        raise RuntimeError("StrongSORT requer: pip install boxmot") from exc
    reid_weights = r"E:\Projeto\Modelos\osnet_x0_25_msmt17.pt"   # string, não Path
    return ("strongsort", StrongSort(reid_weights, device="cpu", fp16=False))


_TRACKER_FACTORIES = {
    "bytetrack":  lambda fps: _criar_bytetrack(fps),
    "botsort":    lambda fps: _criar_botsort(),
    "ocsort":     lambda fps: _criar_ocsort(),
    "deepsort":   lambda fps: _criar_deepsort(),
    "strongsort": lambda fps: _criar_strongsort(),
    "norfair":    lambda fps: ("norfair", NorfairAdapter()),
}


def criar_tracker(config, fps):
    tipo = config["tracker"]
    if tipo not in _TRACKER_FACTORIES:
        raise ValueError(f"Tracker não suportado: {tipo}")
    return _TRACKER_FACTORIES[tipo](fps)


def atualizar_tracker(tracker_info, jogadores: sv.Detections, frame=None) -> sv.Detections:
    modo, obj = tracker_info

    if modo == "supervision":
        return obj.update(jogadores) if len(jogadores) else jogadores

    if modo == "norfair":
        return obj.update(jogadores)

    if modo == "deepsort":
        if len(jogadores) == 0 or frame is None:
            return sv.Detections.empty()
        bbs = [([x1, y1, x2 - x1, y2 - y1], conf, 0)
               for (x1, y1, x2, y2), conf in zip(jogadores.xyxy, jogadores.confidence)]
        tracks = obj.update_tracks(bbs, frame=frame)
        xyxy, tids = [], []
        for t in tracks:
            if not t.is_confirmed():
                continue
            l, t_, r, b = t.to_ltrb()
            xyxy.append([l, t_, r, b])
            tids.append(int(t.track_id))
        if not xyxy:
            return sv.Detections.empty()
        return sv.Detections(xyxy=np.asarray(xyxy, dtype=np.float32),
                             tracker_id=np.asarray(tids, dtype=int),
                             confidence=np.ones(len(xyxy), dtype=np.float32),
                             class_id=np.zeros(len(xyxy), dtype=int))

    if modo == "strongsort":
        if len(jogadores) == 0 or frame is None:
            return sv.Detections.empty()
        dets = np.hstack([jogadores.xyxy, jogadores.confidence.reshape(-1, 1),
                          np.zeros((len(jogadores), 1))])
        saida = obj.update(dets, frame)
        if saida is None or len(saida) == 0:
            return sv.Detections.empty()
        return sv.Detections(xyxy=saida[:, 0:4].astype(np.float32),
                             tracker_id=saida[:, 4].astype(int),
                             confidence=saida[:, 5].astype(np.float32) if saida.shape[1] > 5
                                       else np.ones(len(saida), dtype=np.float32),
                             class_id=np.zeros(len(saida), dtype=int))

    if modo == "ocsort":
        if len(jogadores) == 0:
            return sv.Detections.empty()
        dets = np.hstack([jogadores.xyxy, jogadores.confidence.reshape(-1, 1)]).astype(np.float64)
        saida = obj.update(dets, None)
        if saida is None or len(saida) == 0:
            return sv.Detections.empty()
        saida = np.asarray(saida)
        n_cols = saida.shape[1]
        return sv.Detections(
            xyxy=saida[:, 0:4].astype(np.float32),
            tracker_id=saida[:, n_cols - 1].astype(int),   # ID é sempre a ÚLTIMA coluna
            confidence=np.ones(len(saida), dtype=np.float32),
            class_id=np.zeros(len(saida), dtype=int),)

    if modo == "ocsort_boxmot":
        if len(jogadores) == 0 or frame is None:
            return sv.Detections.empty()
        dets = np.hstack([
            jogadores.xyxy,
            jogadores.confidence.reshape(-1, 1),
            np.zeros((len(jogadores), 1)),   # classe fictícia — não usamos classes aqui
        ]).astype(np.float32)
        saida = obj.update(dets, frame)
        if saida is None or len(saida) == 0:
            return sv.Detections.empty()
        saida = np.asarray(saida)
        return sv.Detections(
            xyxy=saida[:, 0:4].astype(np.float32),
            tracker_id=saida[:, 4].astype(int),
            confidence=saida[:, 5].astype(np.float32) if saida.shape[1] > 5
                        else np.ones(len(saida), dtype=np.float32),
            class_id=np.zeros(len(saida), dtype=int),)


    raise ValueError(f"Modo de tracker desconhecido: {modo}") 


# ============================================================
# DETECTORES
# ============================================================
def carregar_rtdetr_bola(caminho):
    caminho = os.path.abspath(caminho)
    if caminho not in _CACHE_RTDETR_BOLA:
        print(f"Carregando RT-DETR da bola uma vez: {caminho}", flush=True)
        _CACHE_RTDETR_BOLA[caminho] = RTDETR(caminho)
    return _CACHE_RTDETR_BOLA[caminho]


def carregar_detector(config):
    tipo = config["detector"]
    if tipo == "yolo":
        return YOLO(config["model_path"])
    if tipo == "rtdetr":
        return RTDETR(config["model_path"])
    if tipo == "rfdetr":
        try:
            from rfdetr import RFDETRBase
        except ImportError as exc:
            raise RuntimeError("RF-DETR requer: pip install rfdetr") from exc
        return RFDETRBase(pretrain_weights=config["model_path"])
    raise ValueError(f"Detector não suportado: {tipo}")


def detectar(detector, config, frame, largura, altura, classes_override=None):
    """Executa detecção com classes explícitas; evita detectar bola duas vezes."""
    classes = config.get("classes", {})
    player = int(classes.get("player", 0))
    ball = int(classes.get("ball", 32))
    if classes_override is not None:
        ids = {int(x) for x in classes_override}
    else:
        ids = {player, ball}
        if classes.get("goalkeeper") is not None:
            ids.add(int(classes["goalkeeper"]))
        ids.update(set(classes.get("referee", set())))
    imgsz = calcular_imgsz(largura, altura) if config["imgsz"] == "auto" else config["imgsz"]
    tipo = config["detector"]
    if tipo in ("yolo", "rtdetr"):
        resultado = detector(
            frame, conf=min(CONFIANCA_JOGADOR, CONFIANCA_BOLA),
            iou=IOU_DETECCAO, imgsz=imgsz, classes=sorted(ids),
            device=DEVICE, half=HALF_INFERENCIA, verbose=False,
        )[0]
        return sv.Detections.from_ultralytics(resultado)
    if tipo == "rfdetr":
        resultado = detector.predict(frame, threshold=min(CONFIANCA_JOGADOR, CONFIANCA_BOLA))
        if hasattr(resultado, "xyxy"):
            return sv.Detections(
                xyxy=np.asarray(resultado.xyxy, dtype=np.float32),
                confidence=np.asarray(resultado.confidence, dtype=np.float32),
                class_id=np.asarray(resultado.class_id, dtype=int),
            )
        raise RuntimeError("Resultado RF-DETR não reconhecido.")
    raise ValueError(f"Detector não suportado: {tipo}")


def filtrar_especiais(detections, config):
    """Remove classes marcadas como goleiro/árbitro NO NÍVEL DO MODELO
    (o modelo Roboflow já as separa; o COCO não tem essas classes)."""
    if len(detections) == 0:
        return detections
    classes = config.get("classes", {})
    excluir = np.zeros(len(detections), dtype=bool)
    if classes.get("goalkeeper") is not None:
        excluir |= detections.class_id == int(classes["goalkeeper"])
    referees = set(classes.get("referee", set()))
    if referees:
        excluir |= np.isin(detections.class_id, list(referees))
    return detections[~excluir]


def detectar_e_trackear(
    detector_jogadores,
    detector_bola_rtdetr,
    tracker_info,
    config,
    frame,
    largura,
    altura,
    diagnostico=None,
):
    """
    Retorna:
        jogadores_tracked: detecções de jogadores com IDs de tracking
        bola_detections: detecções da bola feitas pelo RT-DETR ou YOLO
    """

    modo = tracker_info[0]

    classes = config.get("classes", {})
    classe_player = int(classes.get("player", 0))
    classe_ball_yolo = int(classes.get("ball", 32))

    imgsz = (
        calcular_imgsz(largura, altura)
        if config.get("imgsz") == "auto"
        else config.get("imgsz")
    )
    t_player = perf_counter()

    # ---------------------------------------------------------
    # 1. Detecção e tracking dos jogadores com YOLO
    # ---------------------------------------------------------
    if modo == "ultralytics_native":
        tracker_yaml = tracker_info[1]

        resultado_jogadores = detector_jogadores.track(
            frame,
            conf=CONFIANCA_JOGADOR,
            iou=IOU_DETECCAO,
            imgsz=imgsz,
            classes=[classe_player],
            device=DEVICE,
            half=HALF_INFERENCIA,
            verbose=False,
            persist=True,
            tracker=tracker_yaml,
        )[0]

        jogadores = sv.Detections.from_ultralytics(resultado_jogadores)

    else:
        detections = detectar(
            detector_jogadores,
            config,
            frame,
            largura,
            altura,
            classes_override=[classe_player],
        )

        filtradas = filtrar_especiais(detections, config)

        jogadores = filtradas[
            filtradas.class_id == classe_player
        ]

        if len(jogadores):
            jogadores = jogadores[
                jogadores.confidence >= CONFIANCA_JOGADOR
            ]

            if len(jogadores):
                jogadores = jogadores.with_nms(
                    threshold=NMS_JOGADORES,
                    class_agnostic=True,
                )

            jogadores = atualizar_tracker(
                tracker_info,
                jogadores,
                frame=frame,
            )
        else:
            jogadores = sv.Detections.empty()

    if diagnostico is not None:
        diagnostico["tempo_deteccao_jogadores"] += perf_counter() - t_player

    # ---------------------------------------------------------
    # 2. Detecção da bola
    # ---------------------------------------------------------
    t_ball = perf_counter()
    if config.get("usar_rtdetr_bola", False):
        if detector_bola_rtdetr is None:
            raise ValueError(
                "usar_rtdetr_bola=True, mas detector_bola_rtdetr não foi fornecido."
            )

        bola = detectar_bola_rtdetr(
            detector_bola_rtdetr,
            frame,
            imgsz,
            classe_ball=config.get("classe_ball_rtdetr", CLASSE_BOLA_RTDETR),
        )
    else:
        # Use YOLO para a bola somente se RT-DETR estiver desativado
        resultado_bola = detector_jogadores(
            frame,
            conf=CONFIANCA_BOLA,
            iou=IOU_DETECCAO,
            imgsz=imgsz,
            classes=[classe_ball_yolo],
            device=DEVICE,
            verbose=False,
        )[0]

        bola = sv.Detections.from_ultralytics(resultado_bola)

    if diagnostico is not None:
        diagnostico["tempo_deteccao_bola"] += perf_counter() - t_ball

    return jogadores, bola



# ============================================================
# SELEÇÃO MANUAL POR PAPÉIS
# ============================================================
_click_pos = None


def _callback(evento, x, y, flags, param):
    global _click_pos
    if evento == cv2.EVENT_LBUTTONDOWN:
        _click_pos = (x, y)


def selecionar_referencias(video_path, detector, config):
    """
    Seleção manual É SÓ GEOMÉTRICA agora: registra role_id + box_referencia.
    Não extrai assinatura aqui — cada teste extrai a sua própria depois,
    a partir do MESMO frame e das MESMAS caixas, evitando incompatibilidade
    de dimensões entre aparências diferentes (HSV=34D, MobileNet=576D, SigLIP=768D).

    Retorna: (refs, frame_referencia)
    """
    global _click_pos
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_REFERENCIA)
    ok, frame = cap.read()
    largura, altura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if not ok:
        raise RuntimeError("Não foi possível ler o frame de referência.")

    classes = config.get("classes", {})
    detections = filtrar_especiais(detectar(detector, config, frame, largura, altura), config)
    detections = detections[detections.class_id == int(classes.get("player", 0))]
    if len(detections) == 0:
        raise RuntimeError("Nenhum jogador detectado no frame de referência.")

    usados, donos, refs = set(), {}, {}
    cores_por_role = {
        r["id"]: cor for r, cor in zip(
            ROLES, [(255, 100, 0), (0, 100, 255), (0, 255, 0),
                    (0, 255, 255), (255, 0, 255), (120, 120, 120)]
        )
    }

    cv2.namedWindow("Selecao", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Selecao", _callback)

    lid = 0
    try:
        for role in ROLES:
            role_id, obrig = role["id"], role.get("obrigatorio", False)
            _click_pos = None
            print(f"[{role['label']}] clique nos jogadores"
                  f"{' (obrigatório)' if obrig else ' — opcional, C sem clicar pula'}"
                  f" | C confirma | ESC cancela.", flush=True)
            contagem_role = 0
            while True:
                tela = frame.copy()
                for idx, box in enumerate(detections.xyxy):
                    x1, y1, x2, y2 = map(int, box)
                    escolhido = idx in usados
                    cor = cores_por_role.get(donos.get(idx), (0, 255, 0)) if escolhido else (0, 255, 0)
                    cv2.rectangle(tela, (x1, y1), (x2, y2), cor, 3 if escolhido else 2)
                    if escolhido:
                        cv2.putText(tela, remover_acentos_cv2(donos[idx][:8]), (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
                cv2.rectangle(tela, (0, 0), (tela.shape[1], 55), (0, 0, 0), -1)
                cv2.putText(tela, remover_acentos_cv2(
                    f"{role['label']}: {contagem_role} | clique | C=confirmar | ESC=cancelar"),
                    (15, 38), cv2.FONT_HERSHEY_SIMPLEX, .65, cores_por_role[role_id], 2)
                cv2.imshow("Selecao", tela)
                tecla = cv2.waitKey(50) & 0xFF

                if tecla == 27:
                    cv2.destroyWindow("Selecao")
                    return {}, None
                if tecla in (ord('c'), ord('C')):
                    if contagem_role > 0 or not obrig:
                        break
                    print("Papel obrigatório — selecione pelo menos um jogador.", flush=True)

                if _click_pos is None:
                    continue
                x, y = _click_pos
                _click_pos = None
                idx = next((i for i, b in enumerate(detections.xyxy)
                           if b[0] <= x <= b[2] and b[1] <= y <= b[3]), None)
                if idx is None or idx in usados:
                    continue

                usados.add(idx)
                donos[idx] = role_id
                refs[lid] = {"logical_id": lid, "role_id": role_id,
                            "box_referencia": [float(v) for v in detections.xyxy[idx]]}
                contagem_role += 1
                print(f"logical_id={lid} -> {role['label']}", flush=True)
                lid += 1
    finally:
        cv2.destroyWindow("Selecao")

    return refs, frame  # frame é devolvido para reextração por teste

# ============================================================
# ASSOCIAÇÃO DE TRACKS
# ============================================================
def _score_aparencia_papel(classificador, tid, role_id):
    if classificador is None or role_id is None:
        return 0.0
    scores = classificador.scores_por_papel(int(tid))
    if not scores:
        return 0.0
    # Similaridade [-1, 1] convertida para [0, 1].
    return float(np.clip((scores.get(role_id, -1.0) + 1.0) / 2.0, 0.0, 1.0))


def associar_tracks(jogadores, refs, estado, largura, altura, frame_num, classificador=None):
    candidatos = []
    for tid, box in zip(jogadores.tracker_id, jogadores.xyxy):
        tid = int(tid)
        for lid, ref in refs.items():
            info = estado[lid]
            base = info.get("ultima_box", np.asarray(ref["box_referencia"], dtype=np.float32))
            iou = iou_boxes(box, base)
            dist = distancia_normalizada(box, base, largura, altura)
            # Predição simples de posição da referência baseada na velocidade histórica.
            vel = np.asarray(info.get("velocidade", [0.0, 0.0]), dtype=np.float32)
            centro_prev = centro_box(base) + vel
            centro_atual = centro_box(box)
            dist_vel = float(np.linalg.norm(centro_atual - centro_prev) / max(1.0, np.hypot(largura, altura)))
            score_vel = max(0.0, 1.0 - min(1.0, dist_vel / max(MATCH_DISTANCIA_MAX, 1e-6)))
            score_dist = max(0.0, 1.0 - min(1.0, dist / max(MATCH_DISTANCIA_MAX, 1e-6)))
            score_app = _score_aparencia_papel(classificador, tid, ref.get("role_id"))
            pred_role, _ = classificador.prever(tid) if classificador is not None else (None, 0.0)
            score_role = 1.0 if pred_role == ref.get("role_id") else 0.0
            if dist > MATCH_DISTANCIA_MAX and iou < MATCH_IOU_MIN and score_app < 0.65:
                continue
            score = (
                PESO_ASSOC_IOU * iou
                + PESO_ASSOC_DIST * score_dist
                + PESO_ASSOC_VEL * score_vel
                + PESO_ASSOC_APARENCIA * score_app
                + PESO_ASSOC_PAPEL * score_role
            )
            candidatos.append((score, tid, lid, iou, dist, box, score_app, score_role))
    candidatos.sort(reverse=True, key=lambda x: x[0])
    usados_tid, usados_lid, pares = set(), set(), []
    for score, tid, lid, iou, dist, box, score_app, score_role in candidatos:
        if tid in usados_tid or lid in usados_lid or score < MATCH_SCORE_MIN:
            continue
        usados_tid.add(tid)
        usados_lid.add(lid)
        info = estado[lid]
        anterior = np.asarray(info.get("ultima_box", box), dtype=np.float32)
        info["velocidade"] = (centro_box(box) - centro_box(anterior)).astype(np.float32)
        info["ultima_box"] = np.asarray(box, dtype=np.float32)
        info["ultimo_frame"] = frame_num
        info["tracker_ids"].add(tid)
        pares.append((tid, lid, score, iou, dist, box, score_app, score_role))
    return pares


def registrar_id(info, tid, frame_num):
    seq = info["sequencia_tracker_ids"]
    if not seq or seq[-1] != tid:
        if seq:
            info["frames_troca_id"].append(frame_num)
        seq.append(tid)


def finalizar_presenca(info, frame_num):
    ultimo = info.get("ultimo_frame_associado")
    if ultimo is not None:
        gap = frame_num - ultimo - PULAR_FRAMES
        if gap > 0:
            info["maior_gap_frames"] = max(info["maior_gap_frames"], gap)
            if gap >= 3 * PULAR_FRAMES:
                info["fragmentacoes"] += 1
    info["ultimo_frame_associado"] = frame_num

def calcular_metricas_troca(info, fps):          # ← ADICIONAR AQUI
    frames = int(info["frames_vistos"])
    trocas = len(info["frames_troca_id"])
    if trocas == 0:
        return {"frames_por_troca": None, "trocas_por_1000_frames": 0.0,
                "intervalo_medio_trocas_segundos": None}
    intervalos = np.diff(info["frames_troca_id"])
    intervalo = float(np.mean(intervalos) * PULAR_FRAMES / fps) if len(intervalos) else frames * PULAR_FRAMES / fps
    return {"frames_por_troca": frames / trocas, "trocas_por_1000_frames": 1000 * trocas / max(1, frames),
           "intervalo_medio_trocas_segundos": intervalo}

def salvar_csv(path, linhas):
    if not linhas:
        return
    campos = sorted({k for linha in linhas for k in linha})
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(linhas)


# ============================================================
# BOLA — estado, continuidade e predição curta
class BallState:
    def __init__(self, fps=30.0):
        self.fps = float(fps or 30.0)
        self.frames_vistos = 0
        self.perdas = 0
        self.frames_perdida_atual = 0
        self.maior_falha = 0
        self.confiancas = []
        self.last_box = None
        self.last_center = None
        self.velocity = np.zeros(2, dtype=np.float32)
        self.last_seen_frame = None
        self.predicted_frames = 0
        self.trajectory = []

    def _centro(self, box):
        return centro_box(box).astype(np.float32)

    def update(self, detections, frame_num):
        if len(detections) > 0:
            idx = int(np.argmax(detections.confidence))
            box = np.asarray(detections.xyxy[idx], dtype=np.float32)
            center = self._centro(box)
            if self.last_center is not None and self.last_seen_frame is not None:
                delta_frames = max(1, frame_num - self.last_seen_frame)
                medida = (center - self.last_center) / delta_frames
                self.velocity = 0.7 * self.velocity + 0.3 * medida
            self.last_box = box
            self.last_center = center
            self.last_seen_frame = frame_num
            self.frames_vistos += 1
            self.frames_perdida_atual = 0
            self.predicted_frames = 0
            self.confiancas.append(float(detections.confidence[idx]))
            self.trajectory.append({"frame": int(frame_num), "x": float(center[0]), "y": float(center[1]), "predicted": False})
            return detections

        if self.last_seen_frame is None:
            return detections
        self.frames_perdida_atual += 1
        if self.frames_perdida_atual == 1:
            self.perdas += 1
        self.maior_falha = max(self.maior_falha, self.frames_perdida_atual)
        self.predicted_frames += 1
        if self.predicted_frames > BALL_MAX_PREDICT_FRAMES:
            return detections

        pred_center = self.last_center + self.velocity * self.predicted_frames
        if self.last_box is None:
            return detections
        box = self.last_box.copy()
        half = (box[2:4] - box[:2]) / 2.0
        pred_box = np.array([
            pred_center[0] - half[0], pred_center[1] - half[1],
            pred_center[0] + half[0], pred_center[1] + half[1],
        ], dtype=np.float32)
        self.trajectory.append({"frame": int(frame_num), "x": float(pred_center[0]), "y": float(pred_center[1]), "predicted": True})
        return sv.Detections(
            xyxy=pred_box.reshape(1, 4),
            confidence=np.array([0.0], dtype=np.float32),
            class_id=np.array([0], dtype=int),
        )

    def as_dict(self):
        return {
            "frames_vistos": self.frames_vistos,
            "perdas": self.perdas,
            "frames_perdida_atual": self.frames_perdida_atual,
            "maior_falha": self.maior_falha,
            "confiancas": self.confiancas,
            "predicted_frames": self.predicted_frames,
            "trajectory_points": len(self.trajectory),
        }


def criar_estado_bola(fps=30.0):
    return BallState(fps=fps)


def atualizar_estado_bola(bola_detections, estado_bola, frame_num):
    return estado_bola.update(bola_detections, frame_num)


# ============================================================
# MÉTRICAS — quebradas por grupo (times, goleiros, árbitro) + bola
# ============================================================
def calcular_metricas(refs, estado, estado_bola, total, fps, processados, inicio, config, imgsz_resolvido, diagnostico=None):
    resumo = []
    por_grupo = defaultdict(list)
    coberturas, presencas, trocas_total = [], [], 0

    for lid, ref in refs.items():
        role_id = ref["role_id"]
        rotulo, grupo, excluido = rotulo_grupo(role_id)
        info = estado[lid]
        vistos = info["frames_vistos"]
        primeiro, ultimo = info["primeiro_frame"], info["ultimo_frame"]
        esperado = max(1, ultimo - primeiro + 1) if primeiro is not None and ultimo is not None else 1
        cobertura = vistos / esperado
        presenca = vistos / max(1, processados)
        troca = calcular_metricas_troca(info, fps)
        trocas = len(info["frames_troca_id"])

        # ── Confiabilidade da classificação de papel/time ──────────
        margem_media = float(np.mean(info["margens_classificacao"])) if info["margens_classificacao"] else 0.0
        taxa_confirmacao = info["frames_confirmados"] / vistos if vistos > 0 else 0.0
        taxa_divergencia = info["frames_divergentes"] / vistos if vistos > 0 else 0.0

        item = {
            "logical_id": lid, "papel": role_id, "rotulo": rotulo, "grupo_metrica": grupo,
            "excluido": excluido, "frames_vistos": vistos,
            "primeiro_frame": primeiro or "", "ultimo_frame": ultimo or "",
            "tempo_visto_segundos": vistos * PULAR_FRAMES / fps, "cobertura": cobertura, "presenca": presenca,
            "trocas_de_tracker_id": trocas, "frames_por_troca": troca["frames_por_troca"],
            "trocas_por_1000_frames": troca["trocas_por_1000_frames"],
            "intervalo_medio_trocas_segundos": troca["intervalo_medio_trocas_segundos"],
            "fragmentacoes": info["fragmentacoes"], "maior_gap_frames": info["maior_gap_frames"],
            "tracker_ids": str(sorted(info["tracker_ids"])),
            "score_medio": float(np.mean(info["scores"])) if info["scores"] else 0.0,
            "margem_media_classificacao": round(margem_media, 4),
            "taxa_confirmacao_papel": round(taxa_confirmacao, 4),
            "taxa_divergencia_papel": round(taxa_divergencia, 4),
        }
        resumo.append(item)

        if not excluido:
            por_grupo[grupo].append(item)
            coberturas.append(cobertura)
            presencas.append(presenca)
            trocas_total += trocas

    # ── Agregação da confiabilidade de classificação (todo o teste) ─
    itens_validos = [x for x in resumo if not x["excluido"]]
    todas_margens       = [x["margem_media_classificacao"] for x in itens_validos]
    todas_confirmacoes  = [x["taxa_confirmacao_papel"]      for x in itens_validos]
    todas_divergencias  = [x["taxa_divergencia_papel"]      for x in itens_validos]

    confiabilidade_out = {
        "margem_media_geral":     round(float(np.mean(todas_margens)), 4)      if todas_margens else 0.0,
        "taxa_confirmacao_media": round(float(np.mean(todas_confirmacoes)), 4) if todas_confirmacoes else 0.0,
        "taxa_divergencia_media": round(float(np.mean(todas_divergencias)), 4) if todas_divergencias else 0.0,
    }

    elapsed = time.time() - inicio
    por_grupo_out = {
        grupo: {
            "jogadores": len(v),
            "cobertura_media": float(np.mean([x["cobertura"] for x in v])),
            "presenca_media": float(np.mean([x["presenca"] for x in v])),
            "trocas": sum(x["trocas_de_tracker_id"] for x in v),
            "frames_vistos": sum(x["frames_vistos"] for x in v),
        }
        for grupo, v in por_grupo.items()
    }

    total_proc = max(1, processados)
    bola_out = {
        "frames_vistos_bola": estado_bola.frames_vistos,
        "presenca_bola": round(estado_bola.frames_vistos / total_proc, 4),
        "num_perdas_bola": estado_bola.perdas,
        "maior_falha_frames": estado_bola.maior_falha,
        "confianca_media_bola": (
            round(float(np.mean(estado_bola.confiancas)), 4)
            if estado_bola.confiancas else 0.0
        ),
    }

    metricas = {
        "nome_teste": config["nome_teste"],
        "detector": config["detector"], "modelo_usado": os.path.basename(config["model_path"]),
        "tracker": config["tracker"], "aparencia": config["aparencia"],
        "rtdetr_bola": bool(config.get("usar_rtdetr_bola", False)),
        "modelo_bola_usado": os.path.basename(config.get("rtdetr_bola_model_path", "")),
        "imgsz_config": config["imgsz"],
        "imgsz": imgsz_resolvido,
        "fps_processamento": processados / elapsed if elapsed > 0 else 0.0,
        "media_cobertura": float(np.mean(coberturas)) if coberturas else 0.0,
        "media_presenca": float(np.mean(presencas)) if presencas else 0.0,
        "total_trocas_tracker_id": trocas_total,
        "frames_por_troca_global": processados / trocas_total if trocas_total else None,
        "trocas_por_1000_frames_global": 1000 * trocas_total / max(1, processados),
        "por_grupo": por_grupo_out,
        "bola": bola_out,
        "diagnostico": diagnostico or {},
        "confiabilidade_time": confiabilidade_out,
    }
    return metricas, resumo

# ============================================================
# EXECUÇÃO DE UM TESTE
# ============================================================
def construir_ancoras_para_config(config, frame_referencia, refs, extrator):
    """
    A partir do MESMO frame e das MESMAS caixas clicadas, extrai a assinatura
    (HSV, MobileNet, SigLIP, DINOv2...) correspondente à aparência DESTE teste.
    Isso garante que a dimensão da âncora sempre bate com a dimensão das
    assinaturas calculadas durante o loop principal — resolvendo os erros
    de "shapes not aligned".
    """
    banco: dict[str, list] = {r["id"]: [] for r in ROLES}

    for ref in refs.values():
        role_id = ref["role_id"]
        crop = crop_torso(frame_referencia, np.asarray(ref["box_referencia"], dtype=np.float32))
        assin = assinatura_por_opcao(crop, config["aparencia"], extrator)
        if assin is not None:
            banco[role_id].append(assin)

    if config["aparencia"] == "kmeans_hsv":
        return banco  # sub-clustering acontece dentro da própria classe

    if config["aparencia"] == "none":
        return {}

    centroides = {k: (np.mean(np.asarray(v), axis=0) if v else None)
                 for k, v in banco.items()}
    return centroides


def executar_teste(config, refs, frame_referencia):
    nome = config["nome_teste"]
    cap = None        # define ANTES de qualquer coisa que possa lançar exceção
    out_video = None  # idem — evita erro no finally se falhar antes de criar o writer

    try:
        detector = carregar_detector(config)

        detector_bola_rtdetr = None

        if config.get("usar_rtdetr_bola", False):
            caminho_rtdetr = config.get("rtdetr_bola_model_path", MODELO_RTDETR_BOLA)

            if not caminho_rtdetr:
                raise ValueError(
                    "usar_rtdetr_bola=True, mas "
                    "'rtdetr_bola_model_path' não foi informado."
                )

            print(
                f"Carregando RT-DETR para a bola: {caminho_rtdetr}",
                flush=True,
            )

            detector_bola_rtdetr = carregar_rtdetr_bola(caminho_rtdetr)

        extrator = criar_extrator(config["aparencia"])


        ancoras_ou_banco = construir_ancoras_para_config(config, frame_referencia, refs, extrator)

        classificador = criar_classificador(config["aparencia"])
        classificador.configurar_ancoras(ancoras_ou_banco)

        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            raise RuntimeError(f"Não foi possível abrir {VIDEO_PATH}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        tracker_info = criar_tracker(config, fps)

        imgsz_resolvido = (
            calcular_imgsz(largura, altura) if config["imgsz"] == "auto" else config["imgsz"]
        )

        # ── Vídeo anotado — SÓ AGORA fps/largura/altura já existem ──────
        ellipse_annotator = label_annotator = triangle_annotator = None
        if SALVAR_VIDEO or EXIBIR_PREVIEW:
            if SALVAR_VIDEO:
                video_out_path = os.path.join(OUTPUT_DIR, f"video-{nome}.mp4")
                out_video = cv2.VideoWriter(
                    video_out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                    fps / max(1, PULAR_FRAMES), (largura, altura)
                )
                if not out_video.isOpened():
                    print(f"[aviso] Não foi possível abrir VideoWriter para {video_out_path}")
                    out_video = None
            if EXIBIR_PREVIEW or out_video is not None:
                ellipse_annotator, label_annotator, triangle_annotator = criar_anotadores()

        estado = {
            lid: {"frames_vistos": 0, "primeiro_frame": None, "ultimo_frame": None,
                  "ultimo_frame_associado": None,
                  "ultima_box": np.asarray(ref["box_referencia"], dtype=np.float32),
                  "velocidade": np.zeros(2, dtype=np.float32),
                  "tracker_ids": set(), "sequencia_tracker_ids": [], "frames_troca_id": [],
                  "fragmentacoes": 0, "maior_gap_frames": 0, "scores": [],
                  "margens_classificacao": [], "frames_confirmados": 0, "frames_divergentes": 0}
            for lid, ref in refs.items()
        }
        estado_bola = criar_estado_bola(fps=fps)

        linhas = []
        inicio = time.time()
        frame_num = processados = 0
        diagnostico = {
            "frames_com_jogadores": 0, "frames_com_bola": 0,
            "deteccoes_jogadores": 0, "deteccoes_bola": 0,
            "novos_tracks": 0, "novos_tracks_classificados": 0,
            "tempo_deteccao_jogadores": 0.0, "tempo_deteccao_bola": 0.0,
            "tempo_aparencia": 0.0, "tempo_associacao": 0.0,
            "tempo_total_loop": 0.0,
        }
        roles_por_track = {}
        tracks_novos_contados = set()
        proximo_log = 0

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame_num += 1
            if frame_num % max(1, PULAR_FRAMES) != 0:
                continue
            processados += 1
            
            if MAX_FRAMES_PROCESSAMENTO is not None and processados > MAX_FRAMES_PROCESSAMENTO:
                break
            
            t_loop = perf_counter()
            
            jogadores, bola = detectar_e_trackear(detector, detector_bola_rtdetr, tracker_info, config, frame, largura, altura)

            if len(bola) > 0:
                bola = bola[bola.confidence >= CONFIANCA_BOLA]
                if len(bola) > 0:
                    bola = bola.with_nms(threshold=NMS_BOLA, class_agnostic=True)
            atualizar_estado_bola(bola, estado_bola, frame_num)

            # ── Classificação de aparência para TODO jogador rastreado ──────
            # (não só os que bateram com uma referência clicada)
            role_por_tracker_id = {}
            if len(jogadores) and config["aparencia"] != "none":
                crops, tids = [], []
                for tid, box in zip(jogadores.tracker_id, jogadores.xyxy):
                    crop = crop_torso(frame, box)
                    if crop is not None:
                        crops.append(crop)
                        tids.append(int(tid))

                if config["aparencia"] in _EXTRATORES_NEURAIS and extrator is not None:
                    embeddings = extrator.extrair(crops) if crops else []
                    for tid, emb in zip(tids, embeddings):
                        classificador.adicionar(tid, emb)
                else:
                    for tid, crop in zip(tids, crops):
                        classificador.adicionar(tid, assinatura_hsv(crop))

                # Aplica o classificador em TODOS os tracker_ids do frame, não só nos pares
                for tid in jogadores.tracker_id:
                    pred_role, margem = classificador.prever(int(tid))
                    role_final = classificador.atualizar(int(tid), pred_role)
                    role_por_tracker_id[int(tid)] = (role_final, margem)

            pares = associar_tracks(jogadores, refs, estado, largura, altura, frame_num) if len(jogadores) else []

            for tid, lid, score, iou, dist, box, score_app, score_role in pares:   
                info = estado[lid]
                if info["primeiro_frame"] is None:
                    info["primeiro_frame"] = frame_num
                info["ultimo_frame"] = frame_num
                info["frames_vistos"] += 1
                info["scores"].append(float(score))
                registrar_id(info, tid, frame_num)
                finalizar_presenca(info, frame_num)

                # Usa a classificação já calculada acima para TODOS, não recalcula aqui
                role_final, margem = role_por_tracker_id.get(int(tid), (None, 0.0))
                classificacao_confirmada = role_final is not None
                if role_final is None:
                    role_final = refs[lid]["role_id"]

                info["margens_classificacao"].append(float(margem))
                if classificacao_confirmada:
                    info["frames_confirmados"] += 1
                if classificacao_confirmada and role_final != refs[lid]["role_id"]:
                    info["frames_divergentes"] += 1

                rotulo, grupo, excluido = rotulo_grupo(role_final)
                linhas.append({
                    "frame": frame_num, "logical_id": lid, "tracker_id": tid,
                    "papel": role_final, "rotulo": rotulo, "grupo_metrica": grupo,
                    "excluido": excluido, "score": score, "iou": iou, "distancia": dist,
                })

            # ── NOVO: registra também jogadores SEM referência (não clicados) ──
            # Isso captura quem entra depois no vídeo, sem logical_id fixo
            tids_com_par = {p[0] for p in pares}
            for tid in jogadores.tracker_id:
                tid_int = int(tid)
                if tid_int in tids_com_par:
                    continue  # já registrado acima via pares
                role_final, margem = role_por_tracker_id.get(tid_int, (None, 0.0))
                if role_final is None:
                    continue  # ainda sem classificação confiável — não registra
                rotulo, grupo, excluido = rotulo_grupo(role_final)
                linhas.append({
                    "frame": frame_num, "logical_id": None, "tracker_id": tid_int,
                    "papel": role_final, "rotulo": rotulo, "grupo_metrica": grupo,
                    "excluido": excluido, "score": None, "iou": None, "distancia": None,
                })

            # ── Vídeo anotado e preview ao vivo ─────────────────────────
            deve_anotar = (
                (out_video is not None or EXIBIR_PREVIEW)
                and frame_num % max(1, PULAR_FRAMES) == 0
            )
            if deve_anotar:
                if len(jogadores) > 0:
                    xyxy_anotar, tids_anotar, classes_anotar, labels_anotar = [], [], [], []

                    # Monta um mapa rápido: tracker_id -> info de quem já tem par (referência clicada)
                    info_por_tid = {}
                    for linha_log in linhas:
                        if linha_log["frame"] == frame_num:
                            info_por_tid[linha_log["tracker_id"]] = linha_log

                    for tid, box in zip(jogadores.tracker_id, jogadores.xyxy):
                        tid_int = int(tid)
                        info_linha = info_por_tid.get(tid_int)

                        if info_linha is not None:
                            role_final_v = info_linha["papel"]
                            rotulo_v = info_linha["rotulo"]
                        else:
                            # Ainda sem classificação confirmada — mostra mesmo assim,
                            # com rótulo neutro, para você ver que já está sendo detectado
                            role_final_v, _ = role_por_tracker_id.get(tid_int, (None, 0.0))
                            rotulo_v = rotulo_grupo(role_final_v)[0] if role_final_v else "?"

                        xyxy_anotar.append(box)
                        tids_anotar.append(tid_int)
                        classes_anotar.append(cor_index_por_papel(role_final_v) if role_final_v else 5)
                        labels_anotar.append(f"#{tid_int} {rotulo_v}")

                    det_anotar = sv.Detections(
                        xyxy=np.array(xyxy_anotar, dtype=np.float32),
                        tracker_id=np.array(tids_anotar, dtype=int),
                        class_id=np.array(classes_anotar, dtype=int),
                    )

                    frame_anotado = ellipse_annotator.annotate(scene=frame.copy(), detections=det_anotar)
                    frame_anotado = label_annotator.annotate(scene=frame_anotado, detections=det_anotar, labels=labels_anotar)
                else:
                    frame_anotado = frame.copy()

                if len(bola) > 0:
                    frame_anotado = triangle_annotator.annotate(scene=frame_anotado, detections=bola)

                cv2.putText(frame_anotado, f"{nome}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame_anotado, f"Frame {frame_num}/{total}", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if out_video is not None:
                    out_video.write(frame_anotado)

                if EXIBIR_PREVIEW and frame_num % max(1, PREVIEW_INTERVALO) == 0:
                    cv2.imshow(f"Tracking — {nome}", frame_anotado)
                    tecla = cv2.waitKey(1) & 0xFF
                    if tecla in (ord("q"), 27):
                        print("Preview interrompido pelo usuário.")
                        break

            diagnostico["tempo_total_loop"] += perf_counter() - t_loop
            if frame_num >= proximo_log:
                imprimir_progresso(nome, frame_num, total, processados, inicio,
                                   f"pares={len(pares)} bola_vista={estado_bola.frames_vistos}")
                proximo_log = frame_num + max(1, int(fps * 5))

        metricas, resumo = calcular_metricas(refs, estado, estado_bola, total, fps, processados, inicio, config, imgsz_resolvido, diagnostico)
        if SALVAR_CSV_DETECCOES:
            salvar_csv(os.path.join(CSV_DIR, nome + "_deteccoes.csv"), linhas)
        if SALVAR_CSV_RESUMO:
            salvar_csv(os.path.join(CSV_DIR, nome + "_resumo.csv"), resumo)
        if SALVAR_JSON_METRICAS:
            with open(os.path.join(JSON_DIR, nome + ".json"), "w", encoding="utf-8") as f:
                json.dump(metricas, f, ensure_ascii=False, indent=2)

        return metricas

    finally:
        if cap is not None:
            cap.release()
        if out_video is not None:
            out_video.release()
        cv2.destroyAllWindows()
# ============================================================
# HTML — resultados anteriores sempre preservados (append, não overwrite);
# detalhe de cada teste abre logo abaixo da própria linha
# ============================================================
def gerar_html(resultados, caminho, nome_video):
    json_path = os.path.join(JSON_DIR, f"resultados_comparacao_{nome_video}.json")

    existentes: dict[str, dict] = {}
    if os.path.isfile(json_path):
        try:
            with open(json_path, encoding="utf-8") as f:
                existentes = {r.get("nome_teste", ""): r for r in json.load(f)}
        except Exception:
            pass

    # ADIÇÃO, não sobrescrita: resultados novos entram/atualizam por nome_teste,
    # tudo o que já estava salvo continua no arquivo.
    for r in resultados:
        existentes[r.get("nome_teste", "")] = r

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(list(existentes.values()), f, ensure_ascii=False, indent=2)

    rows = []
    for i, r in enumerate(existentes.values()):
        nome = html_lib.escape(str(r.get("nome_teste", "")))
        did = f"det_{i}"
        erro = r.get("erro")
        if erro:
            main = f"<td colspan='10' style='color:red'>ERRO: {html_lib.escape(str(erro))}</td>"
        else:
            bola_info = r.get("bola", {})
            confiab = r.get("confiabilidade_time", {})
            main = (
                f"<td>{r.get('imgsz','')}"
                f"{' (auto)' if r.get('imgsz_config')=='auto' else ''}</td>"
                f"<td>{r.get('detector','')}</td><td>{r.get('tracker','')}</td>"
                f"<td>{r.get('modelo_usado','')}</td><td>{r.get('aparencia','')}</td>"
                f"<td>{r.get('fps_processamento',0):.2f}</td><td>{100*r.get('media_cobertura',0):.2f}%</td>"
                f"<td>{r.get('total_trocas_tracker_id',0)}</td>"
                f"<td>{r.get('frames_por_troca_global') if r.get('frames_por_troca_global') else '—'}</td>"
                f"<td>{100*bola_info.get('presenca_bola',0):.1f}%</td>"
                f"<td>{100*confiab.get('taxa_confirmacao_media',0):.1f}%</td>"        # ← nova
                f"<td>{100*confiab.get('taxa_divergencia_media',0):.1f}%</td>"  
                f"<td><button onclick=\"toggle('{did}')\">Detalhes</button></td>"
            )
        detalhe = html_lib.escape(json.dumps(r, ensure_ascii=False, indent=2))
        rows.append(
            f"<tr class='main-row'><td><b>{nome}</b></td>{main}</tr>"
            f"<tr id='{did}' class='detail-row' style='display:none'><td colspan='14'><pre>{detalhe}</pre></td></tr>"
        )

    page = f"""<!doctype html><html><head><meta charset='utf-8'>
<title>Comparação de Tracking — {html_lib.escape(nome_video)}</title>
<style>
body{{font-family:Arial;margin:24px;background:#f9f9f9}}
h1{{color:#2e91ca}}
table{{border-collapse:collapse;width:100%;background:white}}
th,td{{border:1px solid #bbb;padding:8px;text-align:center}}
th{{background:#2e91ca;color:white;cursor:pointer;user-select:none;position:relative}}
th:hover{{background:#1a6fa0}}
th.sorted-desc::after{{content:" ▼";font-size:11px}}
th.sorted-asc::after{{content:" ▲";font-size:11px}}
tr.main-row:nth-child(4n+1){{background:#f5f5f5}}
pre{{text-align:left;white-space:pre-wrap;background:#1e1e1e;color:#d4d4d4;padding:12px}}
button{{background:#2e91ca;color:white;border:0;padding:5px 10px;border-radius:4px;cursor:pointer}}
button:hover{{background:#1a6fa0}}
</style>
<script>
function toggle(id){{
  let x=document.getElementById(id);
  x.style.display=x.style.display==='table-row'?'none':'table-row';
}}

function parseCell(texto) {{
  // remove % e vírgulas de milhar, tenta converter para número
  const limpo = texto.replace('%','').replace(/\\./g,'').replace(',','.').trim();
  const num = parseFloat(limpo);
  return isNaN(num) ? texto.toLowerCase() : num;
}}

function ordenarTabela(colIndex, th) {{
  const table = th.closest('table');
  const tbody = table.querySelector('tbody');
  // Pega só as linhas principais (não as de detalhe)
  const pares = [];
  const linhasPrincipais = tbody.querySelectorAll('tr.main-row');
  linhasPrincipais.forEach(tr => {{
    const detalheId = tr.nextElementSibling; // linha de detalhe logo abaixo
    pares.push([tr, detalheId]);
  }});

  const crescente = th.classList.contains('sorted-asc');
  table.querySelectorAll('th').forEach(h => h.classList.remove('sorted-asc','sorted-desc'));

  pares.sort((a, b) => {{
    const va = parseCell(a[0].children[colIndex].innerText);
    const vb = parseCell(b[0].children[colIndex].innerText);
    if (typeof va === 'number' && typeof vb === 'number') {{
      return crescente ? va - vb : vb - va;
    }}
    return crescente ? String(va).localeCompare(String(vb)) : String(vb).localeCompare(String(va));
  }});

  pares.forEach(([tr, det]) => {{
    tbody.appendChild(tr);
    if (det) tbody.appendChild(det);
  }});

  th.classList.add(crescente ? 'sorted-desc' : 'sorted-asc');
}}

document.addEventListener('DOMContentLoaded', () => {{
  document.querySelectorAll('th').forEach((th, idx) => {{
    if (th.innerText.trim() === 'Detalhes') return;  // não ordena a coluna de botões
    th.addEventListener('click', () => ordenarTabela(idx, th));
  }});
}});
</script>
</head><body>
<h1>Comparação de Tracking — {html_lib.escape(nome_video)}</h1>
<table>
<thead>
<tr>
  <th>Teste</th><th>imgsz</th><th>Detector</th><th>Tracker</th><th>Modelo</th><th>Aparência</th>
  <th>FPS</th><th>Cobertura</th><th>Trocas ID</th><th>Frames/Troca</th><th>Presença Bola</th>
  <th>Confirmação Papel</th><th>Divergência Papel</th><th>Detalhes</th>
</tr>
</thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
</body></html>"""

    with open(caminho, "w", encoding="utf-8") as f:
        f.write(page)


# ============================================================
# MAIN
# ============================================================


def main():
    criar_pastas()
    if not TESTES:
        raise RuntimeError("Lista TESTES está vazia.")

    print("Carregando detector de referência...", flush=True)
    detector_ref = carregar_detector(TESTES[0])

    print("Abrindo seleção manual de papéis (times, goleiros, árbitro, excluir)...", flush=True)
    refs, frame_referencia = selecionar_referencias(VIDEO_PATH, detector_ref, TESTES[0])
    if not refs or frame_referencia is None:
        raise RuntimeError("Nenhuma referência selecionada. Encerrando.")

    print(f"\n{len(refs)} referências selecionadas.", flush=True)

    resultados = []
    for i, config in enumerate(TESTES, 1):
        print(f"\n{'='*60}\nTESTE {i}/{len(TESTES)}: {config['nome_teste']}\n{'='*60}", flush=True)
        try:
            resultados.append(executar_teste(config, refs, frame_referencia))
            print(f"✓ Concluído: {config['nome_teste']}", flush=True)
        except Exception as exc:
            resultados.append({
                "nome_teste": config["nome_teste"], "erro": str(exc),
                "detector": config.get("detector", ""), "video": os.path.basename(VIDEO_PATH),
            })
            print(f"✗ Falhou: {config['nome_teste']} → {exc}", flush=True)

    nome_video_sem_ext = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
    html_path = os.path.join(HTML_DIR, f"relatorio_{nome_video_sem_ext}.html")
    gerar_html(resultados, html_path, nome_video_sem_ext)
    print(f"\nRelatório salvo em: {html_path}", flush=True)


if __name__ == "__main__":
    main()
