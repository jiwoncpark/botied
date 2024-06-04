num_iters = {1: 80+1, 2: 40+1, 4: 20+1}


pretty_acq_names = {
    'pes': 'PES',
    'mes': 'MES',
    'jes': 'JES',
    'botied_copula_cdf': r'BOtied v1',
    'botied_mvn_cdf': r'BOtied$_{\rm MVN}$ v1',
    'botied_empirical_cdf': r'BOtied$_{\rm empirical}$ v1',
    'botied_kde_cdf': r'BOtied$_{\rm KDE}$ v1',
    'botied_copula_inverse_cdf': r'BOtied$_{\rm copula}$ v1',
    'botied_copula_inverse_cdf_of_means': r'BOtied$_{\rm copula}$ v2',
    'botied_copula_cdf_of_means': r'BOtied$_{\rm copula}$ v2',
    'botied_mvn_cdf_of_means': r'BOtied$_{\rm MVN}$ v2',
    'botied_empirical_cdf_of_means': r'BOtied$_{\rm empirical}$ v2',
    'botied_kde_cdf_of_means': r'BOtied v2',  # r'BOtied$_{\rm KDE}$ v2',
    'random': r'Random',
    'qnehvi': r'qNEHVI',
    'qnparego': r'qNParEGO'
}


acq_to_color = {
    'botied_copula_cdf': "#882255",
    'botied_mvn_cdf': "#882255",
    'botied_empirical_cdf': "#882255",
    'botied_kde_cdf': "#882255",
    'botied_copula_inverse_cdf': "#882255",
    'botied_copula_inverse_cdf_of_means': "#AA4499",
    'botied_copula_cdf_of_means': "#AA4499",
    'botied_mvn_cdf_of_means': "#AA4499",
    'botied_empirical_cdf_of_means': "#AA4499",
    'botied_kde_cdf_of_means': "#AA4499",
    'pes': "#33BBEE",
    'mes': "#999933",
    'jes': "#DDCC77",
    'random': "k",
    'qnehvi': "#EE7733",  # "#999933",
    'qnparego': "#117733",
}


acq_to_linewidth = {
    'botied_copula_cdf': 2.0,
    'botied_copula_cdf_of_means': 2.0,
    'botied_mvn_cdf_of_means': 2.0,
    'botied_empirical_cdf_of_means': 2.0,
    'botied_kde_cdf_of_means': 2.0,
    'botied_mvn_cdf': 2.0,
    'botied_copula_inverse_cdf':  2.0,
    'botied_copula_inverse_cdf_of_means': 2.0,
    "pes": 2.0,
    "mes": 2.0,
    "jes": 2.0,
    'random': 2.0,
    'qnehvi': 2.0,
    'qnparego': 2.0}


acq_to_marker = {
    'botied_copula_cdf': 'None',
    'botied_copula_inverse_cdf': 'None',
    'botied_copula_inverse_cdf_of_means': 'None',
    'botied_copula_cdf_of_means': 'None',
    'botied_mvn_cdf_of_means': 'None',
    'botied_empirical_cdf_of_means': 'None',
    'botied_kde_cdf_of_means': 'None',
    'botied_mvn_cdf': 'None',
    "pes": None,
    "mes": None,
    "jes": None,
    'random': None,
    'qnehvi': None,
    'qnparego': None}


acq_to_linestyle = {
    'botied_copula_cdf': "solid",
    'botied_mvn_cdf': "solid",
    'botied_empirical_cdf': "solid",
    'botied_kde_cdf': "solid",
    'botied_copula_inverse_cdf': 'solid',
    'botied_copula_inverse_cdf_of_means': 'solid',
    'botied_copula_cdf_of_means': "solid",
    'botied_mvn_cdf_of_means': "solid",
    'botied_empirical_cdf_of_means': "solid",
    'botied_kde_cdf_of_means': "solid",
    "pes": (0, (3, 5, 1, 5, 1, 5)),
    "mes": (0, (3, 5, 1, 5, 1, 5)),
    "jes": (0, (3, 5, 1, 5, 1, 5)),
    'random': "dotted",
    'qnehvi': "dashdot",
    'qnparego': "dashed",
}


default_acq_names = [
    'botied_copula_cdf',
    'botied_copula_cdf_of_means',
    # 'botied_mvn_cdf_of_means',
    # 'botied_empirical_cdf_of_means',
    # 'botied_kde_cdf_of_means',
    'random', 'qnehvi', 'qnparego']
