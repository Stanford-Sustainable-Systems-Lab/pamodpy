from datetime import datetime, timedelta
from dateutil import tz

import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar


def generate_p_elec(rate_name, time_init, dt, num_days, start_hour):
    datetime_init = datetime.fromtimestamp(time_init, tz=tz.gettz('America/Los_Angeles'))
    datetime_final = datetime_init + timedelta(days=num_days)

    day_init = datetime_init.date()
    day_final = datetime_final.date()
    total_days = (day_final - day_init).days + 1

    p_elec_energy_total = np.zeros(int(np.round(total_days * 24 / dt)))
    p_elec_demand_dict = {
        "peak": np.zeros(int(np.round(total_days * 24 / dt))),
        "part_peak": np.zeros(int(np.round(total_days * 24 / dt))),
        "any_time": np.zeros(int(np.round(total_days * 24 / dt)))
    }

    #Bev-2-S
    if rate_name == "Bev 2 S":

        energy_rates = {
            "peak": 0.39949,
            "off-peak": 0.18626,
            "super off-peak": 0.16299
        }

        demand_rate = np.round(95.56 / 50, decimals=5)

        hours = {
            "peak": [(16, 21)],
            "off-peak": [(21, 24), (0, 9), (14, 16)],
            "super off-peak": [(9, 24)]
        }
        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=day_init, end=day_final).to_pydatetime()
        current_day = day_init
        for day in range(total_days):
            p_elec_energy_day = np.zeros(int(np.round(24 / dt)))
            for (interval_name, intervals) in hours.items():
                for interval in intervals:
                    p_elec_energy_day[int(np.round(interval[0] / dt)): int(np.round(interval[1] / dt))] = \
                        energy_rates[interval_name]
                p_elec_energy_total[int(np.round(day * 24 / dt)):int(np.round((day + 1) * 24 / dt))] = p_elec_energy_day
                p_elec_demand_day_interval = np.full(int(np.round(24 / dt)), demand_rate)
                p_elec_demand_dict[interval_name][int(np.round(day * 24 / dt)):int(np.round((day + 1) * 24 / dt))] = \
                    p_elec_demand_day_interval

            current_day += timedelta(days=1)

    # NYC SC9 Rate III
    elif rate_name == "SC9 Rate III":
        summer_months = (6, 7, 8, 9)  # Jun 1 to Sep 30
        winter_months = (10, 11, 12, 1, 2, 3, 4, 5)  # Oct 1 to May 31

        energy_rates = {
            "summer": {
                "peak": 0.79,
                "part_peak": 0.79,
                "off_peak": 0.79
            },
            "winter": {
                "part_peak": 0.79,
                "off_peak": 0.79
            }
        }

        demand_rates = {
            "summer": {
                "peak": 22.66,
                "part_peak": 10.56,
                "any_time": 21.67
            },
            "winter": {
                "part_peak": 14.67,
                "any_time": 6.21
            }
        }

        hours = {
            "summer": {
                "workday": {
                    "peak": [(8, 22)],
                    "part_peak": [(8, 18)],
                    "off_peak": [(0, 24)]
                },
                "non-workday": {
                    "off_peak": [(0, 24)]
                }
            },
            "winter": {
                "workday": {
                    "part_peak": [(8, 22)],
                    "off_peak": [(0, 24)]
                },
                "non-workday": {
                    "off_peak": [(0, 24)]
                }
            },
        }

        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=day_init, end=day_final).to_pydatetime()
        current_day = day_init
        for day in range(total_days):
            if current_day.month in summer_months:
                period = "summer"
            else:
                period = "winter"
            if current_day.weekday() in (0, 1, 2, 3, 4) and current_day not in holidays:
                day_type = "workday"
            else:
                day_type = "non-workday"

            p_elec_energy_day = np.zeros(int(np.round(24 / dt)))
            for (interval_name, intervals) in hours[period][day_type].items():
                for interval in intervals:
                    p_elec_energy_day[int(np.round(interval[0] / dt)): int(np.round(interval[1] / dt))] = \
                        energy_rates[period][interval_name]
            p_elec_energy_total[int(np.round(day * 24 / dt)):int(np.round((day + 1) * 24 / dt))] = p_elec_energy_day

            for (interval_name, demand_rate) in demand_rates[period].items():
                if interval_name == "any_time":
                    p_elec_demand_day_interval = np.ones(int(np.round(24 / dt))) * demand_rate
                elif interval_name in hours[period][day_type].keys():
                    p_elec_demand_day_interval = np.zeros(int(np.round(24 / dt)))
                    intervals = hours[period][day_type][interval_name]
                    for interval in intervals:
                        p_elec_demand_day_interval[int(np.round(interval[0] / dt)): int(np.round(interval[1] / dt))] = \
                            demand_rate
                else:
                    p_elec_demand_day_interval = np.zeros(int(np.round(24 / dt)))
                p_elec_demand_dict[interval_name][int(np.round(day * 24 / dt)):int(np.round((day + 1) * 24 / dt))] = \
                    p_elec_demand_day_interval

            current_day += timedelta(days=1)

    # NYC SC9 Rate III + SmartCharge Incentive
    elif rate_name == "SC9 Rate III with SC Incentive":
        summer_months = (6, 7, 8, 9)  # Jun 1 to Sep 30
        winter_months = (10, 11, 12, 1, 2, 3, 4, 5)  # Oct 1 to May 31

        energy_rates = {
            "summer": {
                "peak": 0.79,
                "part_peak": 0.79,
                "off_peak": 0.79,
                "off_peak_inc": -0.1
            },
            "winter": {
                "part_peak": 0.79,
                "off_peak": 0.79,
                "off_peak_inc": -0.1
            }
        }

        demand_rates = {
            "summer": {
                "peak": 22.66,
                "part_peak": 10.56,
                "any_time": 21.67
            },
            "winter": {
                "part_peak": 14.67,
                "any_time": 6.21
            }
        }

        hours = {
            "summer": {
                "workday": {
                    "peak": [(8, 22)],
                    "part_peak": [(8, 18)],
                    "off_peak": [(0, 24)],
                    "off_peak_inc": [(0, 8)]
                },
                "non-workday": {
                    "off_peak": [(0, 24)],
                    "off_peak_inc": [(0, 8)]
                }
            },
            "winter": {
                "workday": {
                    "part_peak": [(8, 22)],
                    "off_peak": [(0, 24)],
                    "off_peak_inc": [(0, 8)]
                },
                "non-workday": {
                    "off_peak": [(0, 24)],
                    "off_peak_inc": [(0, 8)]
                }
            },
        }

        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=day_init, end=day_final).to_pydatetime()
        current_day = day_init
        for day in range(total_days):
            if current_day.month in summer_months:
                period = "summer"
            else:
                period = "winter"
            if current_day.weekday() in (0, 1, 2, 3, 4) and current_day not in holidays:
                day_type = "workday"
            else:
                day_type = "non-workday"

            p_elec_energy_day = np.zeros(int(np.round(24 / dt)))
            for (interval_name, intervals) in hours[period][day_type].items():
                for interval in intervals:
                    p_elec_energy_day[int(np.round(interval[0] / dt)): int(np.round(interval[1] / dt))] = \
                        energy_rates[period][interval_name]
            p_elec_energy_total[int(np.round(day * 24 / dt)):int(np.round((day + 1) * 24 / dt))] = p_elec_energy_day

            for (interval_name, demand_rate) in demand_rates[period].items():
                if interval_name == "any_time":
                    p_elec_demand_day_interval = np.ones(int(np.round(24 / dt))) * demand_rate
                elif interval_name in hours[period][day_type].keys():
                    p_elec_demand_day_interval = np.zeros(int(np.round(24 / dt)))
                    intervals = hours[period][day_type][interval_name]
                    for interval in intervals:
                        p_elec_demand_day_interval[int(np.round(interval[0] / dt)): int(np.round(interval[1] / dt))] = \
                            demand_rate
                else:
                    p_elec_demand_day_interval = np.zeros(int(np.round(24 / dt)))
                p_elec_demand_dict[interval_name][int(np.round(day * 24 / dt)):int(np.round((day + 1) * 24 / dt))] = \
                    p_elec_demand_day_interval

            current_day += timedelta(days=1)

    # NYC SmartCharge Incentives
    elif rate_name == "SmartCharge Incentives":

        energy_rate = {
            "off-peak": 0.1,
            "peak": 0
        }

        hours = {
            "off-peak": [(0, 8)],
            "peak": [(9, 24)]
        }

        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=day_init, end=day_final).to_pydatetime()
        current_day = day_init
        for day in range(total_days):

            p_elec_energy_day = np.zeros(int(np.round(24 / dt)))
            for (interval_name, intervals) in hours.items():
                for interval in intervals:
                    p_elec_energy_day[int(np.round(interval[0] / dt)): int(np.round(interval[1] / dt))] = energy_rate[interval_name]
                p_elec_energy_total[int(np.round(day * 24 / dt)):int(np.round((day + 1) * 24 / dt))] = p_elec_energy_day

                p_elec_demand_day_interval = np.zeros(int(np.round(24 / dt)))
                p_elec_demand_dict[interval_name][int(np.round(day * 24 / dt)):int(np.round((day + 1) * 24 / dt))] = p_elec_demand_day_interval

            current_day += timedelta(days=1)

    p_elec_energy = p_elec_energy_total[int(np.round(start_hour / dt)):int(np.round((start_hour + num_days * 24) / dt))]
    for interval_name in p_elec_demand_dict.keys():
        p_elec_demand_dict[interval_name] = p_elec_demand_dict[interval_name][int(np.round(start_hour / dt)):int(
            np.round((start_hour + num_days * 24) / dt))]

    return p_elec_energy, p_elec_demand_dict