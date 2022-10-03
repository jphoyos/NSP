from src.data.data_consolidation import Consolidator
from src.features.basic_features import Featurizer
from src.data.dataset_generator import Splitter
from src.features.history import HistoryCreator
from src.models.nsp_classifier import NspModelDev
from src.visualization.cross_val_viz import CrossValVisualizer
from src.models.validation import StatisticalAnalysis

c = Consolidator(r"data/raw/informe_estadistico/",r"data/interim/Base.csv","data/raw/CITAS_HLCM_no_psw.xlsx")
c.consolidate()

h = HistoryCreator("data/interim/Base.csv")
h.create_history()
h.write_db("data/processed/history.sqlite")

f = Featurizer("data/interim/Base.csv")
f.generate_basic_features()
f.generate_history_feature("data/processed/history.sqlite")
f.write("data/interim/data.csv")

s = Splitter("data/interim/data.csv")
s.split("data/processed/")

d = NspModelDev("data/processed/features_train_resampled.csv","data/processed/label_train_resampled.csv",30000)
d.grid_search("reports/grid_search/",n_jobs=-1)
d.train_models("reports/grid_search/",-1,"reports/cross_val/")
d.train_best_models("models/","reports/grid_search/")
d.predict_best_models("models/","data/processed/features_test.csv","data/processed/label_test.csv")

cross_val_viz = CrossValVisualizer('reports/cross_val/')
cross_val_viz.plot('reports/figures/cross_val.pdf')

statistical_analyzer = StatisticalAnalysis('reports/cross_val/')
statistical_analyzer.analyze()
statistical_analyzer.generate_report('reports/cross_val_statistical_analysis.json')