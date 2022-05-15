# Sources:
#  - https://indico.cern.ch/event/66256/contributions/2071577/attachments/1017176/1447814/EfficiencyErrors.pdf
#  - https://inspirehep.net/files/57287ac8e45a976ab423f3dd456af694
#  - https://arxiv.org/pdf/0908.0130.pdf
#  - http://phys.kent.edu/~smargeti/STAR/D0/Ullrich-Errors.pdf


def compute_variance_bayesian_efficiency(k, n):
    return ((k + 1) / (n + 2) * (k + 2) / (n + 3)) - ((k + 1) ** 2 / (n + 2) ** 2)


def compute_mode_bayesian_efficiency(k, n):
    return k / n


def compute_mean_bayesian_efficiency(k, n):
    return (k + 1) / (n + 2)
