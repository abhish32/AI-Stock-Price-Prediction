import numpy as np

from arthaseer.backtest import backtest
from arthaseer.data import synthetic_prices
from arthaseer.features import make_dataset
from arthaseer.model import ArthaSeerModel


def test_features_have_no_lookahead():
    df = synthetic_prices(600)
    X, y, _ = make_dataset(df, horizon=1)
    # Truncating future data must not change earlier feature rows
    X2, _, _ = make_dataset(df.iloc[:500], horizon=1)
    common = X2.index.intersection(X.index)
    np.testing.assert_allclose(X.loc[common].values, X2.loc[common].values)


def test_train_predict_roundtrip(tmp_path):
    df = synthetic_prices(700)
    m = ArthaSeerModel("TEST", horizon=1).fit(df)
    res = m.predict_next(df)
    assert res["low_price"] < res["predicted_price"] < res["high_price"]
    assert res["signal"] in {"BUY", "SELL", "HOLD"}
    p = str(tmp_path / "m.joblib")
    m.save(p)
    assert ArthaSeerModel.load(p).predict_next(df)["predicted_price"] == res["predicted_price"]


def test_forecast_and_backtest():
    df = synthetic_prices(700)
    m = ArthaSeerModel("TEST").fit(df)
    fc = m.forecast_path(df, 5)
    assert len(fc) == 5 and (fc["low"] <= fc["high"]).all()
    bt = backtest(m.cv_predictions)
    assert "sharpe" in bt["strategy"]


def test_web_api():
    import server

    c = server.app.test_client()
    assert c.get("/api/health").json["status"] == "ok"
    assert c.get("/api/predict?ticker=bad%20t").status_code == 400
    r = c.get("/api/predict?ticker=DEMO&days=3")
    assert r.status_code == 200
    d = r.json
    assert d["prediction"]["signal"] in {"BUY", "SELL", "HOLD"}
    assert len(d["forecast"]["mid"]) == 3
    assert c.get("/").status_code == 200
