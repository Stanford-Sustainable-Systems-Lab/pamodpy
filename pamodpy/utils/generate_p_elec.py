from datetime import datetime, timedelta
from dateutil import tz
from itertools import repeat

import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar


def generate_p_elec(rate_name, time_init, dt, num_days, start_hour):
    datetime_init = datetime.fromtimestamp(time_init, tz=tz.gettz('America/Los_Angeles'))
    datetime_final = datetime_init + timedelta(days=num_days)

    day_init = datetime_init.date()
    day_final = datetime_final.date()
    total_days = (day_final - day_init).days + 1

    p_elec_energy_total = np.zeros(int(np.round(total_days * 24 / dt)))
    p_elec_demand_dict = {}

    #BEV-2-S
    if rate_name == "BEV-2-S":

        energy_rates = {
            "peak": 0.39971,
            "off-peak": 0.18648,
            "super_off-peak": 0.16321
        }

        demand_rates = {
            "any_time": np.round(95.56 / 50, decimals=5)
        }

        hours = {
            "peak": [(16, 21)],
            "off-peak": [(21, 24), (0, 9), (14, 16)],
            "super_off-peak": [(9, 14)],
            "any_time": [(0, 24)]
        }

        p_elec_demand_dict = dict(zip(demand_rates.keys(), repeat(np.zeros(int(np.round(total_days * 24 / dt))))))

        current_day = day_init
        for day in range(total_days):
            p_elec_energy_day = np.zeros(int(np.round(24 / dt)))
            for (interval_name, energy_rate) in energy_rates.items():
                if interval_name in hours.keys():
                    intervals = hours[interval_name]
                    for interval in intervals:
                        p_elec_energy_day[int(np.round(interval[0] / dt)): int(np.round(interval[1] / dt))] = \
                            energy_rates[interval_name]
            p_elec_energy_total[int(np.round(day * 24 / dt)):int(np.round((day + 1) * 24 / dt))] = p_elec_energy_day

            for (interval_name, demand_rate) in demand_rates.items():
                p_elec_demand_day_interval = np.zeros(int(np.round(24 / dt)))
                if interval_name in hours.keys():
                    intervals = hours[interval_name]
                    for interval in intervals:
                        p_elec_demand_day_interval[int(np.round(interval[0] / dt)): int(np.round(interval[1] / dt))] = \
                            demand_rate
                p_elec_demand_dict[interval_name][int(np.round(day * 24 / dt)):int(np.round((day + 1) * 24 / dt))] = \
                    p_elec_demand_day_interval

            current_day += timedelta(days=1)

    elif rate_name == "E-19 Secondary Voltage":
        summer_months = (5, 6, 7, 8, 9, 10)     # May 1 to Oct 31
        winter_months = (11, 12, 1, 2, 3, 4)    # Nov 1 to Apr 30

        energy_rates = {
            "summer": {
                "peak": 0.15098,
                "part_peak": 0.15098,
                "off_peak": 0.14501
            },
            "winter": {
                "part_peak": 0.14242,
                "off_peak": 0.14171
            }
        }

        demand_rates = {
            "summer": {
                "peak": 19.84,
                "part_peak": 16.36,
                "any_time": 34.09
            },
            "winter": {
                "part_peak": 0,
                "any_time": 34.09
            }
        }

        hours = {
            "summer": {
                "workday": {
                    "peak": [(12, 18)],
                    "part_peak": [(8.5, 12), (18, 21.5)],
                    "off_peak": [(21.5, 24), (0, 8.5)],
                    "any_time": [(0, 24)]
                },
                "non-workday": {
                    "off_peak": [(0, 24)],
                    "any_time": [(0, 24)]
                }
            },
            "winter": {
                "workday": {
                    "part_peak": [(8.5, 21.5)],
                    "off_peak": [(21.5, 24), (0, 8.5)],
                    "any_time": [(0, 24)]
                },
                "non-workday": {
                    "off_peak": [(0, 24)],
                    "any_time": [(0, 24)]
                }
            },
        }

        cal = USFederalHolidayCalendar()
        holidays = cal.holidays(start=day_init, end=day_final).to_pydatetime()
        holidays = [holiday.date() for holiday in holidays]
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

            for interval_name in demand_rates[period].keys():
                if interval_name not in p_elec_demand_dict.keys():
                    p_elec_demand_dict[interval_name] = np.zeros(int(np.round(total_days * 24 / dt)))
            p_elec_energy_day = np.zeros(int(np.round(24 / dt)))

            for (interval_name, energy_rate) in energy_rates[period].items():
                if interval_name in hours[period][day_type].keys():
                    intervals = hours[period][day_type][interval_name]
                    for interval in intervals:
                        p_elec_energy_day[int(np.round(interval[0] / dt)): int(np.round(interval[1] / dt))] = \
                            energy_rates[period][interval_name]
            p_elec_energy_total[int(np.round(day * 24 / dt)):int(np.round((day + 1) * 24 / dt))] = p_elec_energy_day

            for (interval_name, demand_rate) in demand_rates[period].items():
                p_elec_demand_day_interval = np.zeros(int(np.round(24 / dt)))
                if interval_name in hours[period][day_type].keys():
                    intervals = hours[period][day_type][interval_name]
                    for interval in intervals:
                        p_elec_demand_day_interval[int(np.round(interval[0] / dt)): int(np.round(interval[1] / dt))] = \
                            demand_rate
                p_elec_demand_dict[interval_name][int(np.round(day * 24 / dt)):int(np.round((day + 1) * 24 / dt))] = \
                    p_elec_demand_day_interval

            current_day += timedelta(days=1)

    # NYC SC9 Rate III
    # https://www.coned.com/en/our-energy-future/electric-vehicles/best-electric-delivery-rate-for-your-charging-station
    # https://lite.coned.com/_external/cerates/documents/elecPSC10/electric-tariff.pdf
    elif rate_name == "SC9 Rate III":
        summer_months = (6, 7, 8, 9)  # Jun 1 to Sep 30
        winter_months = (10, 11, 12, 1, 2, 3, 4, 5)  # Oct 1 to May 31

        energy_rates = {
            "summer": {
                "any_time": 0.0079
            },
            "winter": {
                "any_time": 0.0079
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
                    "any_time": [(0, 24)]
                },
                "non-workday": {
                    "any_time": [(0, 24)]
                }
            },
            "winter": {
                "workday": {
                    "part_peak": [(8, 22)],
                    "any_time": [(0, 24)]
                },
                "non-workday": {
                    "any_time": [(0, 24)]
                }
            },
        }

        current_day = day_init
        for day in range(total_days):
            if current_day.month in summer_months:
                period = "summer"
            else:
                period = "winter"
            if current_day.weekday() in (0, 1, 2, 3, 4):
                day_type = "workday"
            else:
                day_type = "non-workday"

            for interval_name in demand_rates[period].keys():
                if interval_name not in p_elec_demand_dict.keys():
                    p_elec_demand_dict[interval_name] = np.zeros(int(np.round(total_days * 24 / dt)))
            p_elec_energy_day = np.zeros(int(np.round(24 / dt)))

            for (interval_name, energy_rate) in energy_rates[period].items():
                if interval_name in hours[period][day_type].keys():
                    intervals = hours[period][day_type][interval_name]
                    for interval in intervals:
                        p_elec_energy_day[int(np.round(interval[0] / dt)): int(np.round(interval[1] / dt))] = \
                            energy_rates[period][interval_name]
            p_elec_energy_total[int(np.round(day * 24 / dt)):int(np.round((day + 1) * 24 / dt))] = p_elec_energy_day

            for (interval_name, demand_rate) in demand_rates[period].items():
                p_elec_demand_day_interval = np.zeros(int(np.round(24 / dt)))
                if interval_name in hours[period][day_type].keys():
                    intervals = hours[period][day_type][interval_name]
                    for interval in intervals:
                        p_elec_demand_day_interval[int(np.round(interval[0] / dt)): int(np.round(interval[1] / dt))] = \
                            demand_rate
                p_elec_demand_dict[interval_name][int(np.round(day * 24 / dt)):int(np.round((day + 1) * 24 / dt))] = \
                    p_elec_demand_day_interval

            current_day += timedelta(days=1)

    # NYC SC9 Rate III + SmartCharge Incentive
    # https://lite.coned.com/_external/cerates/documents/elecPSC10/electric-tariff.pdf
    # https://www.coned.com/en/save-money/rebates-incentives-tax-credits/rebates-incentives-tax-credits-for-residential-customers/electric-vehicle-rewards
    elif rate_name == "SC9 Rate III with SmartCharge Incentive":
        summer_months = (6, 7, 8, 9)  # Jun 1 to Sep 30
        winter_months = (10, 11, 12, 1, 2, 3, 4, 5)  # Oct 1 to May 31

        energy_rates = {
            "summer": {
                "any_time_excl_incentive": 0.0079,
                "smart_charge_incentive": 0.0079-0.1
            },
            "winter": {
                "any_time_excl_incentive": 0.0079,
                "smart_charge_incentive": 0.0079-0.1
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
                    "any_time": [(0, 24)],
                    "any_time_excl_incentive": [(8, 24)],
                    "smart_charge_incentive": [(0, 8)]
                },
                "non-workday": {
                    "any_time": [(0, 24)],
                    "any_time_excl_incentive": [(8, 24)],
                    "smart_charge_incentive": [(0, 8)]
                }
            },
            "winter": {
                "workday": {
                    "part_peak": [(8, 22)],
                    "any_time": [(0, 24)],
                    "any_time_excl_incentive": [(8, 24)],
                    "smart_charge_incentive": [(0, 8)]
                },
                "non-workday": {
                    "any_time": [(0, 24)],
                    "any_time_excl_incentive": [(8, 24)],
                    "smart_charge_incentive": [(0, 8)]
                }
            },
        }

        current_day = day_init
        for day in range(total_days):
            if current_day.month in summer_months:
                period = "summer"
            else:
                period = "winter"
            if current_day.weekday() in (0, 1, 2, 3, 4) and current_day:
                day_type = "workday"
            else:
                day_type = "non-workday"

            for interval_name in demand_rates[period].keys():
                if interval_name not in p_elec_demand_dict.keys():
                    p_elec_demand_dict[interval_name] = np.zeros(int(np.round(total_days * 24 / dt)))
            p_elec_energy_day = np.zeros(int(np.round(24 / dt)))

            for (interval_name, energy_rate) in energy_rates[period].items():
                if interval_name in hours[period][day_type].keys():
                    intervals = hours[period][day_type][interval_name]
                    for interval in intervals:
                        p_elec_energy_day[int(np.round(interval[0] / dt)): int(np.round(interval[1] / dt))] = \
                            energy_rates[period][interval_name]
            p_elec_energy_total[int(np.round(day * 24 / dt)):int(np.round((day + 1) * 24 / dt))] = p_elec_energy_day

            for (interval_name, demand_rate) in demand_rates[period].items():
                p_elec_demand_day_interval = np.zeros(int(np.round(24 / dt)))
                if interval_name in hours[period][day_type].keys():
                    intervals = hours[period][day_type][interval_name]
                    for interval in intervals:
                        p_elec_demand_day_interval[int(np.round(interval[0] / dt)): int(np.round(interval[1] / dt))] = \
                            demand_rate
                p_elec_demand_dict[interval_name][int(np.round(day * 24 / dt)):int(np.round((day + 1) * 24 / dt))] = \
                    p_elec_demand_day_interval

            current_day += timedelta(days=1)

    p_elec_energy = p_elec_energy_total[int(np.round(start_hour / dt)):int(np.round((start_hour + num_days * 24) / dt))]
    for interval_name in p_elec_demand_dict.keys():
        p_elec_demand_dict[interval_name] = p_elec_demand_dict[interval_name][int(np.round(start_hour / dt)):int(
            np.round((start_hour + num_days * 24) / dt))]

    return p_elec_energy, p_elec_demand_dict