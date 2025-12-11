
from flask import json


class MockModelClient:
    """
    A mock model client that simulates responses from a language model for testing purposes.
    """
    def generate_content(self, prompt: str) -> str:
        print(f"\n--- Mock Client received prompt ---\n{prompt[:200]}...\n-----------------------------------")
        
        # Simple logic to change response based on prompt content
        if "coupling strength" in prompt.lower():
             response = {
                "design_name": "16-Qubit Flux Tunable Array (Revised)",
                "description": "Revised design with stronger coupling.",
                "qubit_count": 16,
                "topology_type": "square",
                "grid_dimensions": [4, 4],
                "qubit_families": [
                    {
                        "family_id": "A", 
                        "frequency_mean": 4.5e9, 
                        "anharmonicity": -220e6, 
                        "asymmetry_max": 0.25, 
                        "charging_energy": 250e6,
                        "drive_line": True,
                        "drive_coupling": 50e6,
                        "readout_resonator": True,
                        "resonator_coupling": 80e6
                    },
                    {
                        "family_id": "B", 
                        "frequency_mean": 5.2e9, 
                        "anharmonicity": -220e6, 
                        "asymmetry_max": 0.25, 
                        "charging_energy": 250e6,
                        "drive_line": True,
                        "drive_coupling": 50e6,
                        "readout_resonator": True,
                        "resonator_coupling": 80e6
                    }
                ],
                "coupling_strength_mean": 150e6, # Changed value
                "coloring_strategy": "RelaxedColoring",
                "design_guidelines": ["Ensure nearest-neighbor coupling only."]
            }
        else:
            response = {
                "design_name": "16-Qubit Flux Tunable Array",
                "description": "A 4x4 square lattice of flux-tunable transmon qubits.",
                "qubit_count": 16,
                "topology_type": "square",
                "grid_dimensions": [4, 4],
                "qubit_families": [
                    {
                        "family_id": "A", 
                        "frequency_mean": 4.5e9, 
                        "anharmonicity": -220e6, 
                        "asymmetry_max": 0.25, 
                        "charging_energy": 250e6,
                        "drive_line": True,
                        "drive_coupling": 40e6,
                        "readout_resonator": True,
                        "resonator_coupling": 70e6
                    },
                    {
                        "family_id": "B", 
                        "frequency_mean": 5.2e9, 
                        "anharmonicity": -220e6, 
                        "asymmetry_max": 0.25, 
                        "charging_energy": 250e6,
                        "drive_line": True,
                        "drive_coupling": 40e6,
                        "readout_resonator": True,
                        "resonator_coupling": 70e6
                    }
                ],
                "coupling_strength_mean": 120e6,
                "coloring_strategy": "RelaxedColoring",
                "design_guidelines": ["Ensure nearest-neighbor coupling only."]
            }
        return json.dumps(response)
