def calculate_molecule_numbers(
    box_size,
    density,
    salt_concentration,
    salt_molar_mass,
    solvent1_molar_mass,
    solvent2_molar_mass,
    solvent_mass_ratio,
):
    NA = 6.02214076e23
    volume = (box_size * 1e-9) ** 3
    salt_moles = salt_concentration * volume * NA
    total_mass_g = density * volume
    salt_mass_g = salt_moles * salt_molar_mass / NA
    solvent_total_mass_g = total_mass_g - salt_mass_g
    solvent1_mass_g = solvent_total_mass_g * (
        solvent_mass_ratio / (1 + solvent_mass_ratio)
    )
    solvent2_mass_g = solvent_total_mass_g / (1 + solvent_mass_ratio)
    solvent1_molecules = solvent1_mass_g / solvent1_molar_mass * NA
    solvent2_molecules = solvent2_mass_g / solvent2_molar_mass * NA
    return {
        "Lithium Salt Molecules": int(salt_moles),
        "Solvent 1 Molecules": int(solvent1_molecules),
        "Solvent 2 Molecules": int(solvent2_molecules),
    }


box_size = 45
density = 1500
salt_concentration = 0.6
salt_molar_mass = 151.91
solvent1_molar_mass = 86.09
solvent2_molar_mass = 73.09
solvent_mass_ratio = 1 / 9

result = calculate_molecule_numbers(
    box_size,
    density,
    salt_concentration,
    salt_molar_mass,
    solvent1_molar_mass,
    solvent2_molar_mass,
    solvent_mass_ratio,
)
print("Molecule counts:")
for key, value in result.items():
    print(f"{key}: {value}")
