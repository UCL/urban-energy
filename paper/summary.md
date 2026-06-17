# NEPI — what we measured, how, and what we found

## The question

We are testing a reframing: that a neighbourhood should be judged not by how much energy it consumes, but by how much access that energy buys — the everyday life a household can reach for the energy it spends. The idea is Jane Jacobs's: a place is efficient *because* of its complexity, the way a rainforest folds a unit of energy through many exchanges and does far more with it, while a desert, for an equivalent amount of energy, does less because it cannot fold it and the energy simply streams through and is lost. We put this on two measured quantities — energy spent and access gained — and the rate between them, across roughly 178,000 English neighbourhoods (Census Output Areas), comparing those where the dominant dwelling is flats against those where it is detached houses.

## Energy: the units, and why

We measure household energy in kilowatt-hours per household (and per person) per year, combining heat with car travel. The heat is *metered, not modelled*: DESNZ's actual gas and electricity, aggregated from postcode to neighbourhood, rather than the modelled SAP ratings behind a building's EPC, which over-predict what homes actually burn (the performance gap).

## Heat

A detached neighbourhood uses about **1.5 times** a flat's heat per household, **1.31 times** per person. The gap is the form's, in three parts: detached homes are bigger, hold more people, and have a leakier shape (more exposed wall, no shared party walls). At the same floor area, occupancy, age and income, the shape alone accounts for about 15%; the rest is the larger homes and households that low density brings.

## How we measured car travel

We want all the car energy a home's location forces, not just the commute — the commute is only about a sixth of car miles, so using it alone undercounts driving roughly sixfold. No open dataset measures all local driving per neighbourhood, so we build it by *constrained disaggregation*. We start from a measured total: the National Travel Survey (NTS9904) gives car-driver miles per person by 2021 rural-urban class — how far the average person drives in a dense city, a town, a village. We then distribute each class's total down to its neighbourhoods using two local signals from the Census, car ownership and commute distance, so lower-ownership and shorter-commute places receive fewer miles. Each class's population-weighted mean is held to the survey figure, so the totals stay as measured and only their distribution across neighbourhoods is estimated. Finally we convert miles to energy using the local fleet's energy per mile, allowing for the share of electric versus petrol cars (DVLA). Car-travel energy runs from about 3,240 kWh per household per year for a flat to about 9,272 for a detached home, about **2.9 times** as much, and makes up 24–37% of all household energy.

| energy, per household / year | Flat | Detached | ratio, per household | ratio, per person |
| --- | --: | --: | --: | --: |
| heat (metered gas + electricity) | 10,194 | 15,020 | 1.5× | 1.31× |
| car travel | 3,240 | 9,272 | 2.9× | 2.46× |
| **total** | 13,674 | 23,832 | **1.74×** | **1.54×** |

The Flat and Detached columns are per household; per person, detached households are about a sixth larger, so every ratio runs lower. The rows are independent medians, so they need not sum exactly to the total.

![Household energy by dwelling type, metered heat plus car travel: a flat's 13,674 kWh/yr against a detached home's 23,832 (1.74×).](../stats/figures/argument/energy_gradient.png)

## Access: the units we compare

Access is the count of things reachable from a neighbourhood over the *real road network*, the distance along the streets people actually use, computed with cityseer over OS Open Roads rather than straight-line. Because it is a property of the *location*, it is the same however the household is counted: per home or per person makes no difference to what is within reach. We count three kinds of thing, each in its own unit: **amenities** (the number of everyday destinations — GPs, pharmacies, hospitals, schools, food outlets, supermarkets, greenspace), **jobs** (the number of jobs reachable, a weighted sum because each workplace carries its own job count), and **people** (the resident population reachable). Each is read at two distances on one ruler — a short walk (1.6 km) and a long drive (25.6 km) — so the on-foot and the drivable figures are directly comparable, and what you reach on foot is a subset of what you reach by car. We use network distance, not straight-line: straight-line over-credits dispersed places, where streets detour most, so the network gap is the wider of the two.

| within reach (median) | Flat, on foot | Flat, 25 km | Detached, on foot | Detached, 25 km | foot | 25 km |
| --- | --: | --: | --: | --: | --: | --: |
| amenities | 209 | 20,812 | 22 | 4,653 | **9.5×** | 4.5× |
| jobs | 6,927 | 807,658 | 598 | 173,447 | **11.6×** | 4.7× |
| people | 17,838 | 2,343,165 | 2,767 | 472,236 | **6.4×** | 5.0× |

The compactness behind it: a flat neighbourhood holds about 79 people per hectare against a detached one's 14, a factor of 5.7.

![Amenities reachable over the network against distance, by dwelling type: a flat reaches several times more at every distance, about 9.5× on foot and 4.5× at a long drive.](../stats/figures/argument/access_curve.png)

## The ratios

Each ratio compares the typical (median) flat-dominant neighbourhood with the typical detached-dominant one. For access the flat is ahead at every distance, and the ratios narrow as distance grows — on foot a flat reaches roughly nine times the amenities, twelve times the jobs and six times the people of a detached neighbourhood; out at a 25 km drive, where a detached home can finally reach into denser places, the flat is still four to five times ahead. For energy the direction reverses: a detached home spends about 1.5 times the heat, 2.9 times the car energy, and 1.74 times the total. The **rate** brings the two together: a flat returns about **2.9 times the access per kilowatt-hour** it spends — or, put the other way, to reach an equivalent basket of amenities a detached home must drive far enough to burn about three times the energy.

![Access per unit of energy, at each neighbourhood's own car catchment: a flat returns about 2.9× the access per kWh of a detached home.](../stats/figures/argument/access_per_kwh.png)

## Lock-in: what survives decarbonisation

Does decarbonisation close the gap? We recompute the energy with each home at best-practice insulation (its EPC-potential fabric) and a fully electric fleet.

| per household / year | Flat | Detached | ratio, per household | ratio, per person |
| --- | --: | --: | --: | --: |
| total energy, now | 13,674 | 23,832 | 1.74× | 1.54× |
| after best insulation + a full electric fleet | 9,788 | 14,420 | 1.47× | 1.30× |

The energy gap closes only part way: per household from 1.74 to 1.47 times, or per person from 1.54 to 1.30 times. Detached households are about a sixth larger (2.4 vs 2.1 people), so per person the gap is smaller, but it holds on either basis — per household is what is consumed and emitted, per person is per resident.

That residual splits across both halves of the form. About 2,575 kWh of it is **build form**: a best-insulated detached home is still bigger, so it still uses about 1.30 times a flat's heat, because insulation fixes heat loss per square metre, not floor area. About 2,119 kWh is **mobility**: electrification cuts the energy per mile but not the miles, so a detached home still drives about 2.9 times as far, electric or petrol. Technology optimises the efficiency of each unit but leaves the structural quantities, floor area and distance, untouched.

The access gap does not move at all, because no amount of insulation and no electric motor brings a school, a job or a shop any closer to a house built far from them. The inefficiency of dispersed form is therefore not something technology retires; it is fixed in the street layout, which changes only when places are physically rebuilt, over generations rather than product cycles. That permanence is the heart of the finding, and the reason access has to be measured and planned for directly.

## The NEPI scorecard, Atlas and models

We will provide the measure as three things: a **NEPI scorecard**, an EPC-style rating for neighbourhoods rather than buildings; an **Atlas** to explore the ratings; and **XGBoost models** that predict a neighbourhood's NEPI from its form, fabric and fleet, so different combinations can be simulated, including a proposed development before it is built.

---

*Checks: gas under-recording (flats record fewer gas meters than households, coverage 0.81 vs 0.94, but the heat gap holds at 1.45× on well-measured neighbourhoods); access measured over the real network, not straight-line; land-use mix excluded (diversity differs little, ~1.1×, so the access advantage is one of quantity, not balance). Reproduce: `stats/oa_network_access.py`, `stats/access_profile.py`, `stats/lock_in.py`, `stats/form_size_decomposition.py`.*
