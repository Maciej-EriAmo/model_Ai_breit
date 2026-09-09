"""
Dane fizyczne pod walidację Most:

1) Problem trzech ciał (hierarchia masy + energia spektralna)
2) Układ Słoneczny (Słońce + planety; opcjonalnie Księżyc)

Content = kinematyka / masa (x).
Tracer  = hierarchia + energia właściwa (obserwator ΔE) — kanon Most.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from config import CANON

G = 1.0


# --- real-ish solar masses & semi-major axes (AU, M_sun) ---
SOLAR = [
    # name, mass_Msun, a_AU, hierarchy: 0 star | 1 rocky | 2 giant
    ("Sun", 1.0, 0.0, 0),
    ("Mercury", 1.66e-7, 0.387, 1),
    ("Venus", 2.45e-6, 0.723, 1),
    ("Earth", 3.00e-6, 1.000, 1),
    ("Mars", 3.23e-7, 1.524, 1),
    ("Jupiter", 9.55e-4, 5.203, 2),
    ("Saturn", 2.86e-4, 9.537, 2),
    ("Uranus", 4.37e-5, 19.19, 2),
    ("Neptune", 5.15e-5, 30.07, 2),
]


@dataclass
class NBodyBatch:
    content: torch.Tensor  # [B,N,d_x]
    tracer: torch.Tensor  # [B,N,n]
    labels: torch.Tensor  # [B]
    meta_energy: torch.Tensor  # [B,N] raw specific energy (debug)


def _specific_energy(m_central: float, r: float, v: float) -> float:
    r = max(r, 1e-6)
    return 0.5 * v * v - G * m_central / r


def _kepler_state(a: float, M: float, phase: float):
    """Circular Kepler around origin (Sun at 0); M unused for v except via a."""
    r = max(a, 1e-6)
    v = math.sqrt(G * 1.0 / r)  # M_sun=1
    x, y = r * math.cos(phase), r * math.sin(phase)
    vx, vy = -v * math.sin(phase), v * math.cos(phase)
    return x, y, vx, vy, v, r


def solar_system_batch(
    batch_size: int,
    d_x: int = CANON.d_x,
    n_dim: int = CANON.n_dim,
    include_moon: bool = True,
    seed: int = 0,
) -> NBodyBatch:
    """
    Tokeny = Słońce + 8 planet (+ Księżyc przy Ziemi).
    Label = interakcje znaków contentu WEWNĄTRZ warstw hierarchii % 4
    (jak tree_data — Most wygrywa gdy tracer = warstwy energii).

    Tracer:
      0..2 one-hot hierarchy (star / rocky / giant)
      3    energia właściwa znorm. (po batchu)
      4    log(a) znorm.
      5    masa log znorm. / mazur-like
    Moon: osobna gałąź kinematyki (offset Ziemi), hier=1 (rocky).
    """
    rng = np.random.default_rng(seed)
    bodies = list(SOLAR)
    if include_moon:
        # Księżyc: warstwa rocky (1), energia bliska Ziemi wyróżnia w tracerze
        bodies.append(("Moon", 3.69e-8, 1.0, 1))

    N = len(bodies)
    content = torch.zeros(batch_size, N, d_x)
    tracer = torch.zeros(batch_size, N, n_dim)
    energy = torch.zeros(batch_size, N)
    labels = torch.zeros(batch_size, dtype=torch.long)
    hierarchy = [b[3] for b in bodies]

    for b in range(batch_size):
        phases = rng.uniform(0, 2 * math.pi, size=N)
        raw_feats = []
        raw_E = []
        raw_a = []
        raw_m = []
        for i, (name, mass, a, hier) in enumerate(bodies):
            if hier == 0:
                x = y = vx = vy = v = r = 0.0
                eps = 0.0
            elif name == "Moon":
                # Księżyc: offset względem Ziemi (Earth index w SOLAR = 3)
                earth_i = 3
                ex, ey, evx, evy, ev, er = _kepler_state(
                    1.0, 1.0, float(phases[earth_i])
                )
                dx, dy = 0.0025 * math.cos(phases[i]), 0.0025 * math.sin(phases[i])
                x, y = ex + dx, ey + dy
                vx, vy = evx, evy
                r = math.hypot(x, y)
                v = math.hypot(vx, vy)
                eps = _specific_energy(1.0, r, v)
            else:
                # rocky (hier=1) i giant (hier=2): Kepler wokół Słońca
                x, y, vx, vy, v, r = _kepler_state(a, mass, float(phases[i]))
                eps = _specific_energy(1.0, r, v)

            # content: losowe cechy + lekka kinematyka zszumiona
            # masa/hierarchia TYLKO w tracerze (inaczej Softmax czyta warstwę z mass)
            kin = torch.tensor([x, y, vx, vy, r, v], dtype=torch.float32)
            kin = kin / (kin.norm() + 1e-6)
            rnd = torch.randn(d_x)
            feat = 0.7 * rnd
            feat[: min(6, d_x)] = feat[: min(6, d_x)] + 0.3 * kin[: min(6, d_x)]
            feat = feat / feat.norm().clamp_min(1e-6)
            raw_feats.append(feat)
            raw_E.append(eps)
            raw_a.append(math.log10(max(a, 1e-6)) if hier != 0 else -2.0)
            raw_m.append(math.log10(max(mass, 1e-12)))

        feats = torch.stack(raw_feats, dim=0)
        content[b] = feats
        energy[b] = torch.tensor(raw_E, dtype=torch.float32)

        # label: jak tree — suma znaków sum contentu w warstwie
        y = 0
        for layer in sorted(set(hierarchy)):
            idxs = [i for i, h in enumerate(hierarchy) if h == layer]
            layer_sum = feats[idxs].sum()
            y = y + (1 if layer_sum > 0 else 0) * (layer + 1)
        labels[b] = y % CANON.n_classes

        # tracer (wstępny; normalizacja E/a/m po batchu)
        for i, hier in enumerate(hierarchy):
            if n_dim >= 3:
                for k in range(3):
                    tracer[b, i, k] = 1.0 if k == min(hier, 2) else 0.0
            if n_dim == 1:
                tracer[b, i, 0] = float(hier)

    # normalizacja energii / a / m w batchu → dims 3,4,5
    e_min, e_max = energy.min().item(), energy.max().item()
    denom = max(e_max - e_min, 1e-6)
    for b in range(batch_size):
        for i, (name, mass, a, hier) in enumerate(bodies):
            if n_dim > 3:
                tracer[b, i, 3] = (energy[b, i] - e_min) / denom
            if n_dim > 4:
                a_log = -2.0 if hier == 0 else math.log10(max(a, 1e-6))
                tracer[b, i, 4] = (a_log + 2.0) / 4.0  # rough [0,1]
            if n_dim > 5:
                m_log = math.log10(max(mass, 1e-12))
                tracer[b, i, 5] = (m_log + 12.0) / 12.0

    return NBodyBatch(content=content, tracer=tracer, labels=labels, meta_energy=energy)


def three_body_batch(
    batch_size: int,
    d_x: int = CANON.d_x,
    n_dim: int = CANON.n_dim,
    seed: int = 0,
) -> NBodyBatch:
    """
    Trzy ciała: Verlet (krótka integracja) + hierarchia = rank masy (0/1/2).
    Etykieta = interakcje znaków contentu WEWNĄTRZ warstw masy % n_classes
    (jak Φ-drzewo; nie indeks uciekiniera).

    Tracer: masa-rank one-hot + energia spektralna (meta); dim5 = log m, nie content.
    """
    rng = np.random.default_rng(seed)
    N = 3
    content = torch.zeros(batch_size, N, d_x)
    tracer = torch.zeros(batch_size, N, n_dim)
    energy = torch.zeros(batch_size, N)
    mass_log = torch.zeros(batch_size, N)
    labels = torch.zeros(batch_size, dtype=torch.long)

    for b in range(batch_size):
        # masses: primary heavy
        m = np.array(
            [
                rng.uniform(0.8, 1.2),
                rng.uniform(0.05, 0.2),
                rng.uniform(0.02, 0.15),
            ],
            dtype=np.float64,
        )
        mass_log[b] = torch.tensor(
            [math.log10(max(float(mi), 1e-12)) for mi in m], dtype=torch.float32
        )
        # positions
        pos = rng.normal(0, 1.0, size=(3, 2))
        pos -= (m[:, None] * pos).sum(0) / m.sum()
        # velocities ~ circular-ish + noise
        vel = rng.normal(0, 0.4, size=(3, 2))
        vel -= (m[:, None] * vel).sum(0) / m.sum()

        def _acc(p):
            a = np.zeros_like(p)
            for i in range(3):
                for j in range(3):
                    if i == j:
                        continue
                    rij = p[j] - p[i]
                    r2 = float(np.dot(rij, rij)) + 1e-6
                    r = math.sqrt(r2)
                    a[i] += G * m[j] * rij / (r2 * r)
            return a

        # short Verlet
        dt = 0.02
        for _ in range(40):
            a0 = _acc(pos)
            vel = vel + 0.5 * dt * a0
            pos = pos + dt * vel
            a1 = _acc(pos)
            vel = vel + 0.5 * dt * a1

        # energy w.r.t. CM at end — escaper = max eps
        cm = (m[:, None] * pos).sum(0) / m.sum()
        cv = (m[:, None] * vel).sum(0) / m.sum()
        eps = []
        for i in range(3):
            r = float(np.linalg.norm(pos[i] - cm)) + 1e-6
            v = float(np.linalg.norm(vel[i] - cv))
            # potential w.r.t others
            pot = 0.0
            for j in range(3):
                if i == j:
                    continue
                rij = float(np.linalg.norm(pos[i] - pos[j])) + 1e-6
                pot -= G * m[j] / rij
            eps.append(0.5 * v * v + pot)
        energy[b] = torch.tensor(eps, dtype=torch.float32)
        # hierarchy by mass rank: heaviest=0, mid=1, lightest=2 — TYLKO tracer
        order = list(np.argsort(-m))
        hier_of = {order[0]: 0, order[1]: 1, order[2]: 2}

        feats = []
        for i in range(3):
            hier = hier_of[i]
            kin = torch.tensor(
                [
                    float(pos[i, 0]),
                    float(pos[i, 1]),
                    float(vel[i, 0]),
                    float(vel[i, 1]),
                    float(np.linalg.norm(pos[i])),
                    float(np.linalg.norm(vel[i])),
                ],
                dtype=torch.float32,
            )
            kin = kin / (kin.norm() + 1e-6)
            rnd = torch.randn(d_x)
            feat = 0.7 * rnd
            feat[: min(6, d_x)] = feat[: min(6, d_x)] + 0.3 * kin[: min(6, d_x)]
            feat = feat / feat.norm().clamp_min(1e-6)
            content[b, i] = feat
            feats.append(feat)
            if n_dim >= 3:
                for k in range(3):
                    tracer[b, i, k] = 1.0 if k == min(hier, 2) else 0.0

        # label jak drzewo: interakcje WEWNĄTRZ warstw masy (nie uciekinier — bez przecieku)
        y = 0
        for layer in (0, 1, 2):
            idxs = [i for i in range(3) if hier_of[i] == layer]
            if not idxs:
                continue
            layer_sum = torch.stack([feats[i] for i in idxs]).sum()
            y = y + (1 if layer_sum > 0 else 0) * (layer + 1)
        labels[b] = y % CANON.n_classes

    e_min, e_max = energy.min().item(), energy.max().item()
    denom = max(e_max - e_min, 1e-6)
    ml_min, ml_max = mass_log.min().item(), mass_log.max().item()
    ml_den = max(ml_max - ml_min, 1e-6)
    for b in range(batch_size):
        for i in range(N):
            if n_dim > 3:
                tracer[b, i, 3] = (energy[b, i] - e_min) / denom
            if n_dim > 4:
                tracer[b, i, 4] = float(i) / 2.0
            if n_dim > 5:
                tracer[b, i, 5] = (mass_log[b, i] - ml_min) / ml_den

    return NBodyBatch(content=content, tracer=tracer, labels=labels, meta_energy=energy)


def make_nbody_loaders(
    kind: str = "solar",
    n_train: int = 80,
    n_val: int = 15,
    batch_size: int = 32,
    **kw,
):
    if kind == "solar":
        gen = solar_system_batch
    elif kind == "three":
        kw.pop("include_moon", None)
        gen = three_body_batch
    else:
        raise ValueError(kind)
    train = [gen(batch_size, seed=1000 + i, **kw) for i in range(n_train)]
    val = [gen(batch_size, seed=9000 + i, **kw) for i in range(n_val)]
    return train, val
