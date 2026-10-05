q = 1.602e-19
k_B = 1.38e-23
NA = 6.02e23

D = 0.1548
print(D)
D_cm2_s = D * 1e-5
box_size_A = 48.4326
num_ions = 23
T = 300
z = 1

D_m2_s = D_cm2_s * 1e-4
box_size_m = box_size_A * 1e-10
volume_m3 = box_size_m**3

if volume_m3 <= 0:
    raise ValueError("Calculated volume is invalid. Please check box size input.")

conductivity = (q**2 * num_ions * D_m2_s * z**2) / (k_B * T * volume_m3)
conductivity_mS_cm = conductivity * 1e3 / 1e2

print(f"Ionic conductivity: {conductivity:.6e} S/m")
print(f"Ionic conductivity: {conductivity_mS_cm:.6f} mS/cm")
