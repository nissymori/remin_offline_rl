from agents.fql import FQLAgent
from agents.fql_remin_det import FQLReMinDetAgent
from agents.fql_remin_ensemble import FQLReMinEnsembleAgent
from agents.ifql import IFQLAgent
from agents.iql import IQLAgent
from agents.rebrac import ReBRACAgent
from agents.rebrac_remin_det import ReBRACReMinDetAgent
from agents.rebrac_remin_ensemble import ReBRACReMinEnsembleAgent
from agents.sac import SACAgent

agents = dict(
    fql=FQLAgent,
    fql_remin_det=FQLReMinDetAgent,
    fql_remin_ensemble=FQLReMinEnsembleAgent,
    ifql=IFQLAgent,
    iql=IQLAgent,
    rebrac=ReBRACAgent,
    rebrac_remin_det=ReBRACReMinDetAgent,
    rebrac_remin_ensemble=ReBRACReMinEnsembleAgent,
    sac=SACAgent,
)
