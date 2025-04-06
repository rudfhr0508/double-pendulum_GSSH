import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# 이중 진자 운동 방정식
def double_pendulum(u, t, m1, m2, L1, L2, g):
    du = np.zeros(4)
    c = np.cos(u[0] - u[2])
    s = np.sin(u[0] - u[2])
    du[0] = u[1]
    du[1] = (m2 * g * np.sin(u[2]) * c - m2 * s * (L1 * c * u[1]**2 + L2 * u[3]**2) 
             - (m1 + m2) * g * np.sin(u[0])) / (L1 * (m1 + m2 * s**2))
    du[2] = u[3]
    du[3] = ((m1 + m2) * (L1 * u[1]**2 * s - g * np.sin(u[2]) + g * np.sin(u[0]) * c) 
             + m2 * L2 * u[3]**2 * s * c) / (L2 * (m1 + m2 * s**2))
    return du

# 물리적 상수
m1 = 1.0
m2 = 1.0
L1 = 1.0
L2 = 1.0
g = 9.8

# 시간 배열
tfinal = 25.0
Nt = 751
t = np.linspace(0, tfinal, Nt)

# 초기 조건
u0 = [-np.pi/2.2, 0, np.pi/1.8, 0]
perturbation = 1e-10
u0_perturbed = u0 + np.array([perturbation, 0, 0, 0])

# 궤적 시뮬레이션
sol1 = odeint(double_pendulum, u0, t, args=(m1, m2, L1, L2, g))
sol2 = odeint(double_pendulum, u0_perturbed, t, args=(m1, m2, L1, L2, g))

# 궤적 간 거리 계산
distance = np.linalg.norm(sol1 - sol2, axis=1)
d0 = np.linalg.norm(u0 - u0_perturbed)

# 리아프노프 지수 추정
log_divergence = np.log(distance / d0)
lambda_est = np.mean(np.gradient(log_divergence, t))

print(f"추정된 리아프노프 지수: {lambda_est}")

# 시각화
plt.plot(t, log_divergence)
plt.xlabel('시간 (s)')
plt.ylabel('로그 발산')
plt.title('궤적의 로그 발산 변화')
plt.grid(True)
plt.show()