# -*- coding: utf-8 -*-
"""Квазиньютоновский (модифицированный метод Ньютона), формула (2.14).

Варианты 1, 2 и 20 из лабораторной работы № 1.
"""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent / "результаты"


def f1(x):
    return -math.exp(-x) * math.log(x)


def f2(x):
    return -math.exp(-x) * x


def f20(x):
    return x / (x * x - 2 * x + 5)


VARIANTS = [
    {
        "n": 1,
        "title": "F(x) = -e^(-x) * ln(x)",
        "f": f1,
        "a": 0.1,
        "b": 3.0,
    },
    {
        "n": 2,
        "title": "F(x) = -e^(-x) * x",
        "f": f2,
        "a": 0.1,
        "b": 3.0,
    },
    {
        "n": 20,
        "title": "F(x) = x / (x^2 - 2x + 5)",
        "f": f20,
        "a": -4.0,
        "b": 2.0,
    },
]

EPSILONS = (0.1, 0.01, 0.001)
H = 1e-4  # малый шаг для разностных производных


def deriv(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


def step_214(f, x, h):
    """Формула (2.14): x - h * (f(x+h)-f(x-h)) / (f(x+h)-2f(x)+f(x-h))."""
    fp_num = f(x + h) - f(x - h)
    fpp_num = f(x + h) - 2 * f(x) + f(x - h)
    if abs(fpp_num) < 1e-18:
        raise ZeroDivisionError("знаменатель формулы (2.14) близок к нулю")
    return x - h * fp_num / fpp_num, fp_num, fpp_num


def optimize(f, x0, eps, a, b, h=H, max_iter=50):
    x = x0
    rows = []
    for k in range(1, max_iter + 1):
        hh = h
        # для ln(x) шаг не должен выйти за x > 0
        while hh > 1e-8:
            try:
                f(x + hh)
                f(x - hh)
                break
            except (ValueError, ZeroDivisionError):
                hh *= 0.5
        x_raw, fp_num, fpp_num = step_214(f, x, hh)
        fp = fp_num / (2 * hh)
        note = ""

        # f''(x_k) < 0 и шаг идёт в сторону роста — разворот
        if fpp_num < 0 and fp * (x_raw - x) > 0:
            x_raw = x - (x_raw - x)
            note = "разворот шага (f''<0)"

        x_new = x_raw
        halves = 0
        while f(x_new) > f(x) + 1e-14 and halves < 30:
            # функция выросла: дробим шаг, формула (x_{k+1}+x_k)/2
            x_new = (x_new + x) / 2
            halves += 1
        if halves:
            extra = f"дробление шага x{halves}"
            note = f"{note}; {extra}" if note else extra

        # удержание на заданном отрезке
        if x_new < a or x_new > b:
            x_new = min(max(x_new, a), b)
            note = (note + "; " if note else "") + "проекция на [a,b]"

        fp_new = deriv(f, x_new, hh if x_new - hh > 1e-8 else max(x_new * 0.5, 1e-8))
        fv = f(x_new)
        crit = abs(fp_new / fv) if abs(fv) > 1e-15 else abs(fp_new)
        rows.append(
            {
                "k": k,
                "x": x,
                "fx": f(x),
                "x_new": x_new,
                "fx_new": fv,
                "crit": crit,
                "note": note,
            }
        )
        x = x_new
        if crit < eps:
            break
    return x, rows


def analytical_check():
    # вариант 1: ln(x) = 1/x
    x = 1.5
    for _ in range(20):
        g = math.log(x) - 1 / x
        gp = 1 / x + 1 / x**2
        x -= g / gp
    x1 = x
    # вариант 2: x = 1
    x2 = 1.0
    # вариант 20: x = -sqrt(5)
    x20 = -math.sqrt(5)
    return {
        1: (x1, f1(x1)),
        2: (x2, f2(x2)),
        20: (x20, f20(x20)),
    }


def main():
    OUT.mkdir(exist_ok=True)
    exact = analytical_check()
    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("Модифицированный метод Ньютона, формула (2.14)")
    emit("x_{k+1} = x_k - h * (f(x_k+h) - f(x_k-h)) / (f(x_k+h) - 2f(x_k) + f(x_k-h))")
    emit(f"Малый шаг h = {H}")
    emit("Начальное приближение x0 = (a + b) / 2")
    emit("Останов: |f'(x_{k+1}) / f(x_{k+1})| < E")
    emit()

    for var in VARIANTS:
        f = var["f"]
        a, b = var["a"], var["b"]
        x0 = (a + b) / 2
        xs, ys = exact[var["n"]]
        emit("=" * 72)
        emit(f"Вариант {var['n']}.  {var['title']}")
        emit(f"Отрезок [{a}, {b}],  x0 = {x0}")
        emit(f"Стационарная точка (проверка): x* = {xs:.10f},  F(x*) = {ys:.10f}")
        emit()

        # график
        grid = np.linspace(a, b, 400)
        values = np.array([f(float(t)) for t in grid])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(grid, values, color="#1f4e79", lw=2)
        ax.scatter([xs], [ys], color="#c0392b", zorder=3, label="минимум")
        ax.axhline(0, color="#bbbbbb", lw=0.8)
        ax.set_title(f"Вариант {var['n']}: {var['title']}")
        ax.set_xlabel("x")
        ax.set_ylabel("F(x)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUT / f"variant_{var['n']}.png", dpi=140)
        plt.close(fig)

        for eps in EPSILONS:
            x_star, rows = optimize(f, x0, eps, a, b)
            emit(f"--- E = {eps} ---")
            emit(
                f"{'k':>3} {'x_k':>14} {'F(x_k)':>14} {'x_(k+1)':>14} "
                f"{'F(x_k+1)':>14} {'|f\'/f|':>12}  примечание"
            )
            for r in rows:
                emit(
                    f"{r['k']:3d} {r['x']:14.8f} {r['fx']:14.8f} {r['x_new']:14.8f} "
                    f"{r['fx_new']:14.8f} {r['crit']:12.6e}  {r['note']}"
                )
            emit(
                f"Результат: x = {x_star:.8f},  F(x) = {f(x_star):.8f},  "
                f"итераций = {len(rows)}"
            )
            emit()

    report = OUT / "расчет.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Сохранено: {report}")


if __name__ == "__main__":
    main()
