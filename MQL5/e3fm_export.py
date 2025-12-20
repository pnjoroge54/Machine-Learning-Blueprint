import numpy as np


def calculate_ef3m_parameters(concurrent_positions):
    """
    Calculate EF3M parameters in Python and export for MQL5
    """
    # Your existing EF3M implementation
    from afml.bet_sizing.ef3m import M2N, moment, most_likely_parameters, raw_moment

    # Calculate moments from concurrent positions
    central_moments = [moment(concurrent_positions, moment=i) for i in range(1, 6)]
    raw_moments = raw_moment(central_moments, np.mean(concurrent_positions))

    # Fit mixture
    m2n = M2N(raw_moments)
    df_results = m2n.mp_fit()
    params = most_likely_parameters(df_results)

    return [
        params["mu_1"],
        params["mu_2"],
        params["sigma_1"],
        params["sigma_2"],
        params["p_1"],
    ]


# Export to file for MQL5 to read
def export_ef3m_parameters(parameters, output_file):
    np.savetxt(output_file, parameters, delimiter=",")
