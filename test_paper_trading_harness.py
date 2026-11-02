import json
import os
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

import paper_trading_harness as h

PRICES = {"VTI": 370.0, "SMH": 520.0, "XLI": 168.0, "XLU": 43.5}
MON_NOV_2 = date(2026, 11, 2)
MON_DEC_7 = date(2026, 12, 7)


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        d = Path(self.dir.name)
        self.cfg = h.Config(input_path=d / "in.json", state_path=d / "state.json", log_path=d / "log.jsonl")

    def tearDown(self):
        self.dir.cleanup()

    def write_prices(self, prices):
        self.cfg.input_path.write_text(json.dumps({"prices": prices}))

    def run_main(self, today, prices=PRICES):
        self.write_prices(prices)
        with mock.patch.dict(os.environ, {"BROKER": "mock"}):
            return h.main(self.cfg, today)

    def state(self):
        return json.loads(self.cfg.state_path.read_text())

    def events(self):
        return [json.loads(line)["event"] for line in self.cfg.log_path.read_text().splitlines()]

    def weights(self, prices):
        broker = h.MockBroker(self.state(), prices, MON_NOV_2, 1.0)
        return h.current_weights(broker, self.cfg.targets)


class TestDriftBand(Base):
    def test_first_run_buys_targets(self):
        self.assertEqual(self.run_main(MON_NOV_2), 0)
        self.assertEqual(self.events(), ["portfolio_initialized", "initial_allocation"])
        for t, w in self.weights(PRICES).items():
            self.assertAlmostEqual(w, h.TARGET_WEIGHTS[t], places=4)
        self.assertAlmostEqual(self.state()["cash"], 0.0, places=6)
        self.assertEqual(self.state()["last_check"], "2026-11-02")

    def test_second_run_same_month_does_nothing(self):
        self.run_main(MON_NOV_2)
        before = self.cfg.state_path.read_text()
        self.run_main(date(2026, 11, 9), {**PRICES, "SMH": 1000.0})
        self.assertEqual(self.events()[-1], "skipped_already_checked_this_month")
        self.assertEqual(self.cfg.state_path.read_text(), before)

    def test_small_drift_next_month_no_trades(self):
        self.run_main(MON_NOV_2)
        lots_before = self.state()["lots"]
        self.run_main(MON_DEC_7, {**PRICES, "SMH": PRICES["SMH"] * 1.3})
        self.assertEqual(self.events()[-1], "within_band")
        self.assertEqual(self.state()["lots"], lots_before)
        self.assertEqual(self.state()["last_check"], "2026-12-07")

    def test_large_drift_rebalances_to_targets(self):
        self.run_main(MON_NOV_2)
        moved = {**PRICES, "SMH": PRICES["SMH"] * 2.2}
        self.run_main(MON_DEC_7, moved)
        self.assertEqual(self.events()[-1], "rebalance")
        for t, w in self.weights(moved).items():
            self.assertAlmostEqual(w, h.TARGET_WEIGHTS[t], places=4)
        self.assertGreater(self.state()["realized_short_term"], 0)
        self.assertEqual(self.state()["realized_long_term"], 0)

    def test_gain_on_lot_older_than_a_year_is_long_term(self):
        self.run_main(MON_NOV_2)
        self.run_main(date(2027, 12, 6), {**PRICES, "SMH": PRICES["SMH"] * 2.2})
        self.assertGreater(self.state()["realized_long_term"], 0)
        self.assertEqual(self.state()["realized_short_term"], 0)

    def test_weekend_skips(self):
        self.run_main(MON_NOV_2)
        before = self.cfg.state_path.read_text()
        self.run_main(date(2026, 12, 5), {**PRICES, "SMH": 2000.0})
        self.assertEqual(self.events()[-1], "skipped_weekend")
        self.assertEqual(self.cfg.state_path.read_text(), before)


class TestSafety(Base):
    def test_missing_price_halts_without_touching_state(self):
        self.assertEqual(self.run_main(MON_NOV_2, {"VTI": 370.0, "SMH": 520.0}), 1)
        self.assertEqual(self.events(), ["halted_bad_prices"])
        self.assertFalse(self.cfg.state_path.exists())

    def test_old_ticker_input_halts(self):
        old = {"TSM": 414.8, "XLK": 181.6, "CAT": 628.1, "XLE": 69.5}
        self.assertEqual(self.run_main(MON_NOV_2, old), 1)

    def test_non_positive_price_halts(self):
        self.assertEqual(self.run_main(MON_NOV_2, {**PRICES, "XLU": 0}), 1)

    def test_real_broker_refused(self):
        self.write_prices(PRICES)
        with mock.patch.dict(os.environ, {"BROKER": "robinhood"}):
            self.assertEqual(h.main(self.cfg, MON_NOV_2), 2)
        self.assertFalse(self.cfg.state_path.exists())

    def test_approval_on_proposes_without_trading(self):
        self.cfg.require_human_approval = True
        self.run_main(MON_NOV_2)
        self.assertEqual(self.events()[-1], "awaiting_approval")
        self.assertEqual(self.state()["cash"], self.cfg.starting_cash)
        self.assertIsNone(self.state()["last_check"])

    def test_momentum_state_is_retired_and_restarted(self):
        self.cfg.state_path.write_text(json.dumps(
            {"cash": 799.5, "positions": {"CAT": 0.2303}, "price_history": {}, "breaker_day": "2026-10-30"}))
        self.run_main(MON_NOV_2)
        self.assertEqual(self.events()[:3], ["momentum_state_retired", "portfolio_initialized", "initial_allocation"])
        self.assertEqual(self.state()["schema"], "drift_band_v1")


if __name__ == "__main__":
    unittest.main()
