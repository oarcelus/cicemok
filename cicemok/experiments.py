import numpy as np
import pandas as pd
from typing import List, Tuple
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.signal import savgol_filter
from pydantic import BaseModel


class Experiment(BaseModel):
    _experiment: pd.DataFrame = pd.DataFrame()

    def from_xlsx(self, filename: str, sheetname: str) -> None:
        self._experiment = pd.read_excel(filename, sheet_name=sheetname)

    def from_csv(self, filename: str) -> None:
        self._experiment = pd.read_csv(filename)

    def get_experiment(self) -> pd.DataFrame:
        return self._experiment

    def set_experiment(self, experiment: pd.DataFrame) -> None:
        self._experiment = experiment

    def plot(self, x: str, y: str, **kwargs):
        self._experiment.plot(x=x, y=y, **kwargs)


class GITTExperiment(Experiment):
    _rests: List[np.ndarray] = []
    _cc: List[np.ndarray] = []

    def build_list(
        self,
        restindex: int,
        ccindex: int,
        voltage: str = "Voltage(V)",
        capacity: str = "Capacity(mAh)",
        time: str = "Seconds",
        index: str = "Step Index",
        current: str = "Current(mA)",
    ) -> None:
        self._rests = []
        self._cc = []
        rest = []
        cc = []
        cumulative_capacity = 0.0
        for i in range(len(self._experiment) - 1):
            if self._experiment[index].iloc[i] == ccindex:
                if self._experiment[current].iloc[i] > 0:
                    cap = cumulative_capacity + self._experiment[capacity].iloc[i]
                else:
                    cap = cumulative_capacity - self._experiment[capacity].iloc[i]

                cc.append(
                    [
                        self._experiment[time].iloc[i],
                        cap,
                        self._experiment[voltage].iloc[i],
                        self._experiment[current].iloc[i],
                    ]
                )

                if self._experiment[index].iloc[i + 1] == restindex:
                    if self._experiment[current].iloc[i] > 0:
                        cumulative_capacity += self._experiment[capacity].iloc[i]
                    else:
                        cumulative_capacity -= self._experiment[capacity].iloc[i]

                    self._cc.append(np.array(cc))
                    cc = []
            elif self._experiment[index].iloc[i] == restindex:
                rest.append(
                    [
                        self._experiment[time].iloc[i],
                        self._experiment[voltage].iloc[i],
                    ]
                )
                if self._experiment[index].iloc[i + 1] == ccindex:
                    self._rests.append(np.array(rest))
                    rest = []

    def analyze_gitt(
        self,
        linear_region: List[Tuple[float, float]],
        voltage: float
    ):
        assert all([region[0] < region[1] for region in linear_region])
        assert len(linear_region) == len(self._cc)

        ks = []
        rs = []
        vs = []
        socs = []
        """ Temporary solution because GITT starts from unrelaxed CC """
        sqrt_time = np.sqrt(self._cc[0][:, 0])

        mask = (sqrt_time[:] >= linear_region[0][0]) & (
            sqrt_time[:] <= linear_region[0][1]
        )
        x = self._cc[0][mask, :]
        model = LinearRegression()
        model.fit(np.sqrt(x[:, 0]).reshape((-1, 1)), x[:, 2])
        yline = model.predict(sqrt_time.reshape((-1, 1)))
        r = (yline[0] - voltage) / self._cc[0][-1, 3]
        k = model.coef_[0] / self._cc[0][-1, 3]

        rs.append(r)
        ks.append(k)
        socs.append(self._cc[0][-1, 1])
        vs.append(voltage)
        """"""
        count = 1
        for rest, cc, region in zip(self._rests[:-1], self._cc[1:], linear_region[1:]):
            count += 1
            sqrt_time = np.sqrt(cc[:, 0])

            mask = (sqrt_time[:] >= region[0]) & (
                sqrt_time[:] <= region[1]
            )
            x = cc[mask, :]
            if x.size == 0:
                continue

            model = LinearRegression()
            model.fit(np.sqrt(x[:, 0]).reshape((-1, 1)), x[:, 2])
            yline = model.predict(sqrt_time.reshape((-1, 1)))

            derivative = np.gradient(cc[:, 2], sqrt_time)
            y_smooth = savgol_filter(derivative, window_length=10, polyorder=2)

            fig, ax = plt.subplots(2)
            ax[0].scatter(sqrt_time, cc[:, 2], s=10)
            ax[0].plot(sqrt_time, yline, color="r")
            ax[1].plot(sqrt_time, derivative, color="g")
            ax[1].plot(sqrt_time, y_smooth, color="y")
            plt.title(f"N {count}")
            plt.show()

            r = (yline[0] - rest[-1, 1]) / cc[-1, 3]
            k = model.coef_[0] / cc[-1, 3]

            rs.append(r)
            ks.append(k)
            socs.append(cc[-1, 1])
            vs.append(rest[-1, 1])

        return ks, rs, socs, vs

    def analyze_auto_gitt(
        self,
        window_size: int,
        window_var: int = 5,
    ):
        ks = []
        rs = []
        vs = []
        socs = []

        for rest, cc in zip(self._rests[:-1], self._cc[1:]):
            sqrt_time = np.sqrt(cc[:, 0])
            derivative = np.gradient(cc[:, 2], sqrt_time)
            running_average = np.convolve(derivative, np.ones(window_size)/window_size, mode="valid")
            df = pd.Series(running_average)
            running_var = df.rolling(window=window_var).var().to_numpy()

            idxmin = np.nanargmin(running_var)

            start = idxmin
            end = idxmin + window_size
            x = cc[start:end, :]
            if x.size == 0:
                continue

            model = LinearRegression()
            model.fit(np.sqrt(x[:, 0]).reshape((-1, 1)), x[:, 2])
            yline = model.predict(sqrt_time.reshape((-1, 1)))

            derivative = np.gradient(cc[:, 2], sqrt_time)
            y_smooth = savgol_filter(derivative, window_length=10, polyorder=2)

            fig, ax = plt.subplots(2)
            ax[0].scatter(sqrt_time, cc[:, 2], s=10)
            ax[0].plot(sqrt_time, yline, color="r")
            ax[1].plot(sqrt_time, derivative, color="g")
            ax[1].plot(sqrt_time, y_smooth, color="y")
            ax[1].plot(sqrt_time[start:end], derivative[start:end], color="r")
            ax[1].plot(sqrt_time[0:len(running_average)], running_average, color="b")
            # plt.plot(sqrt_time, derivative2, color="g")
            # plt.plot(sqrt_time, y_smooth2, color="b")
            plt.show()

            r = (yline[0] - rest[-1, 1]) / cc[-1, 3]
            k = model.coef_[0] / cc[-1, 3]

            rs.append(r)
            ks.append(k)
            socs.append(cc[-1, 1])
            vs.append(rest[-1, 1])

        return ks, rs, socs, vs

    def get_rests(self) -> List[np.ndarray]:
        return self._rests

    def set_rests(self, rests: List[np.ndarray]) -> None:
        self._rests = rests

    def get_cc(self) -> List[np.ndarray]:
        return self._cc

    def set_cc(self, cc: List[np.ndarray]) -> None:
        self._cc = cc


class ICIExperiment(Experiment):
    _rests: List[np.ndarray] = []
    _cc: List[np.ndarray] = []

    def build_list(
        self,
        restindex: int,
        ccindex: int,
        voltage: str = "Voltage(V)",
        capacity: str = "Capacity(mAh)",
        time: str = "Seconds",
        index: str = "Step Index",
        current: str = "Current(mA)",
    ) -> None:
        self._rests = []
        self._cc = []
        rest = []
        cc = []
        cumulative_capacity = 0.0
        for i in range(len(self._experiment) - 1):
            if self._experiment[index].iloc[i] == ccindex:
                if self._experiment[current].iloc[i] > 0:
                    cap = cumulative_capacity + self._experiment[capacity].iloc[i]
                else:
                    cap = cumulative_capacity - self._experiment[capacity].iloc[i]
                cc.append(
                    [
                        self._experiment[time].iloc[i],
                        cap,
                        self._experiment[voltage].iloc[i],
                        self._experiment[current].iloc[i],
                    ]
                )
                if self._experiment[index].iloc[i + 1] == restindex:
                    if self._experiment[current].iloc[i] > 0:
                        cumulative_capacity += self._experiment[capacity].iloc[i]
                    else:
                        cumulative_capacity -= self._experiment[capacity].iloc[i]

                    self._cc.append(np.array(cc))
                    cc = []
            elif self._experiment[index].iloc[i] == restindex:
                rest.append(
                    [
                        self._experiment[time].iloc[i],
                        self._experiment[voltage].iloc[i],
                    ]
                )
                if self._experiment[index].iloc[i + 1] == ccindex:
                    self._rests.append(np.array(rest))
                    rest = []

    def analyze_ici(self, linear_region: Tuple[float, float], with_plots: bool = False):
        assert linear_region[0] < linear_region[1]
        ks = []
        rs = []
        rselec = []
        rsct = []
        vs = []
        socs = []
        for rest, cc in zip(self._rests, self._cc):
            sqrt_time = np.sqrt(rest[:, 0])
            mask = (sqrt_time >= linear_region[0]) & (sqrt_time <= linear_region[1])
            x = rest[mask, :]
            if x.size == 0:
                continue
            model = LinearRegression()
            model.fit(np.sqrt(x[:, 0]).reshape((-1, 1)), x[:, 1])
            yline = model.predict(sqrt_time.reshape((-1, 1)))

            if with_plots:
                fig, ax = plt.subplots()
                ax.scatter(sqrt_time, rest[:, 1], s=10)
                ax.plot(sqrt_time, yline, color="r")
                plt.show()

            r = (cc[-1, 2] - yline[0]) / cc[-1, 3]
            relec = (cc[-1, 2] - rest[0, 1]) / cc[-1, 3]
            rct = (cc[-1, 2] - x[0, 1]) / cc[-1, 3]
            k = -model.coef_[0] / cc[-1, 3]

            rs.append(r)
            rselec.append(relec)
            rsct.append(rct)
            ks.append(k)
            socs.append(cc[-1, 1])
            vs.append(rest[-1, 1])

        return ks, rs, rselec, rsct, socs, vs

    def get_rests(self) -> List[np.ndarray]:
        return self._rests

    def set_rests(self, rests: List[np.ndarray]) -> None:
        self._rests = rests

    def get_cc(self) -> List[np.ndarray]:
        return self._cc

    def set_cc(self, cc: List[np.ndarray]) -> None:
        self._cc = cc
