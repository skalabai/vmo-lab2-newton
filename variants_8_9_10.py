# Формула (2.14) для вариантов 8, 9 и 10.
# x_{k+1} = x_k - h * (f(x+h) - f(x-h)) / (f(x+h) - 2f(x) + f(x-h))

H = 0.0001


def f8(x):
    return x**4 - 12 * x**3 + 7 * x**2 - 5 * x


def f9(x):
    return x**4 - 10 * x**2 + 5 * x


def f10(x):
    return x**4 - 10 * x**3 + 20 * x


def step(f, x):
    return x - H * (f(x + H) - f(x - H)) / (f(x + H) - 2 * f(x) + f(x - H))


def run(f, a, b):
    x = (a + b) / 2
    rows = []
    for k in range(1, 16):
        x2 = step(f, x)
        n = 0
        while f(x2) > f(x) and n < 20:
            x2 = (x + x2) / 2
            n += 1
        if x2 < a or x2 > b:
            return rows, None
        x = x2
        fp = (f(x + H) - f(x - H)) / (2 * H)
        rows.append((k, x, f(x), abs(fp / f(x))))
        if rows[-1][3] < 0.001:
            break
    return rows, x


def main():
    tasks = [
        (8, "x^4 - 12x^3 + 7x^2 - 5x", f8, 8, 9.5),
        (9, "x^4 - 10x^2 + 5x", f9, 6.5, 8.5),
        (10, "x^4 - 10x^3 + 20x", f10, 6.5, 8),
    ]
    print("Формула (2.14), h = 0.0001, x0 = (a + b) / 2\n")
    for n, title, f, a, b in tasks:
        print(f"Вариант {n}.  F(x) = {title},  [{a}, {b}]")
        rows, x = run(f, a, b)
        if x is None:
            left, right = f(a), f(b)
            xm, fm = (a, left) if left < right else (b, right)
            print(f"  На отрезке нет стационарной точки, F возрастает.")
            print(f"  Минимум на отрезке: x = {xm},  F = {fm:.4f}\n")
            continue
        print(f"  {'k':>2}  {'x':>12}  {'F(x)':>12}")
        shown = set()
        for k, xk, fk, crit in rows:
            mark = ""
            for e in (0.1, 0.01, 0.001):
                if crit < e and e not in shown:
                    mark += f"  E={e}"
                    shown.add(e)
            print(f"  {k:2d}  {xk:12.6f}  {fk:12.4f}{mark}")
        print()


if __name__ == "__main__":
    main()
