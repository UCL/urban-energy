# NEPI: energy spent, access gained

The premise under test is a reframing: that a neighbourhood should be judged not by how much energy it consumes, but by how much access that energy buys, the everyday life a household can reach for the energy it spends. The framing draws on Jane Jacobs: ecosystems, and by extension cities, are more efficient when they are compact and complex, because energy then cascades from use to use through many exchanges before it dissipates, doing more work along the way. A rainforest moves a unit of energy through many such cycles and gets far more from it; a desert, given the same energy, gets less, because it cannot hold it and the energy runs straight through. The test rests on two measured quantities, energy spent and access gained, and the rate between them, across roughly 178,000 English neighbourhoods (Census Output Areas), comparing flat-type against detached-type neighbourhoods.

## Method

Every "×" in this document is a flat-versus-detached gap, computed the same way.

- The unit is the Census 2021 Output Area, the smallest area the census publishes, about 300 residents (~125 households). England has roughly 178,000 of them. Wales is left out for now: the energy, EPC, census and OS inputs all cover it, but the deprivation control used here (England's Index of Multiple Deprivation) has no directly comparable Welsh equivalent, so it would need a harmonised Welsh source.
- The whole dwelling mix is used rather than a single label. The census gives each area its full mix, for example 60% flats, 25% terraced, 10% semi-detached, 5% detached. Rather than labelling each area by its most common type, every area's full mix enters one regression (a compositional model), which reads off the energy or access of a pure all-flat and a pure all-detached area. The gap between those two is the reported figure. Using every proportion is sharper than a one-type label.
- The model holds income, tenure and building age constant and weights by the number of households, so the gap reflects the difference the form makes, not differences in wealth, tenure or the age of the stock. Access is the exception (see that section): there compactness is the mechanism, so it is not held constant.
- Both per household and per person are reported. Detached households are about a sixth larger, so a gap per person is smaller than the same gap per household. Per household is what is consumed and emitted; per person is per resident.
- In every table the Flat and Detached columns are observed medians (metered energy, reachable counts), shown to ground the numbers. The ratio columns are the compositional estimate, so they need not equal the quotient of the two columns.
- The model reads the gap at the extremes, a wholly flat area against a wholly detached one, which few real areas are. Each ratio is therefore the sharp end of the estimate: the gap is at least this large.

*The regression has no intercept and is weighted by households; the flat-to-detached ratio is the exponentiated gap between the pure-flat and pure-detached coefficients. The energy axes are fitted with a log model, the access counts with a Poisson model. Standard errors are not adjusted for spatial autocorrelation between neighbouring areas; the effects are large enough on ~178,000 areas that this does not affect the conclusions.*

## Energy

Household energy is measured in kilowatt-hours per household (and per person) per year, in two parts: the home's metered gas and electricity, and car travel. The home part is *metered, not modelled*: DESNZ's actual gas and electricity, not the modelled SAP ratings behind a building's EPC, which over-predict consumption (the performance gap). Most of it is space and water heating; appliances, lighting and cooking are the smaller, steadier remainder.

How the energy figure is built:

- Every figure is delivered energy, the energy that arrives at the home to be used: gas and electricity for the home, fuel or electricity for the car. It is not primary energy or carbon, so no conversion factors enter; the units are kilowatt-hours per year.
- DESNZ publishes metered gas and electricity as two separate datasets at postcode level, each as a meter count and a mean per meter. Each fuel is aggregated to the Output Area by a meter-weighted mean (postcodes with more meters count for more), and the two are then summed as delivered energy (gas kWh plus electricity kWh) before dividing by census households. Gas is weather-corrected, electricity is not, and both cover domestic meters only.
- An EPC rating is a design estimate of how a building should perform; metered consumption is what households actually use. The two diverge (the performance gap), so the metered figure is the more reliable measure of energy spent.

## Heat

A detached neighbourhood uses about 1.41 times a flat's heat per household and 1.08 times per person. The gap has three parts: detached homes are bigger, hold more people, and have a leakier shape (more exposed wall, no shared party walls). With floor area, occupancy, age, income and tenure held equal, the shape alone accounts for about 12% (1.12× per household); the rest is the larger homes and households that low density brings.

| heat, kWh ph/yr | Flat | Terraced | Semi | Detached | ratio ph | ratio pp |
| --- | --: | --: | --: | --: | --: | --: |
| gas + electricity | 10,194 | 12,995 | 13,876 | 15,020 | 1.41× | 1.08× |

Flats record fewer domestic gas meters than households (coverage about 0.81, against 0.94 for detached), for two reasons the measure treats differently. An all-electric flat heats with electricity, which is summed in, so its heat is captured. A block on communal heating is metered as non-domestic, so that gas is genuinely missing and flat heat understated. Only this second, smaller share is a true gap, and it does not drive the result: with gas coverage held constant the gap is 1.38×, and on well-measured areas about 1.6×, so the 1.41× figure is conservative.

Separating shape from size:

- The detached-versus-flat heat gap blends three effects of low density: bigger homes, more people per home, and a leakier shape. The quantity of interest is the part attributable to shape alone.
- From the compositional model (the full dwelling mix, with age, income and tenure held equal), controls are added one at a time, first household size, then floor area, and the gap shrinks. The reduction is the share mediated by size and occupancy, about three-quarters of the gap. What survives once both are held fixed (about 1.12×, roughly 12% per household) is the direct effect of the form: exposed walls and no shared surfaces.
- Local climate (heating-degree-days, from HadUK-Grid) is the one confound still to add; it would refine the direct term further.

*Reproduce: `stats/form_size_decomposition.py` (the shape-versus-size ladder and the gas-coverage checks).*

## Car travel

The target quantity is the total car energy associated with a home's location, not only the commute. The commute is about a sixth of car miles, so a commute-only figure understates driving by roughly sixfold. Total local driving per neighbourhood is not available directly from open data, so it is built by constrained disaggregation, starting from a measured total: the National Travel Survey (NTS9904) gives car-driver miles per person by 2021 rural-urban class, the average distance driven in a dense city, a town, or a village. Each class's total is distributed to its neighbourhoods using two local signals from the Census, car ownership and commute distance, so lower-ownership and shorter-commute places receive fewer miles. Each class's population-weighted mean is held to the survey figure, so the totals stay as measured and only their distribution across neighbourhoods is estimated. Miles are then converted to energy using the local fleet's energy per mile, allowing for the share of electric versus petrol cars (DVLA).

Building the estimate:

- Total driving by the residents of one neighbourhood is not recorded in open data. The census records the journey to work but not other car travel, and that journey is about a sixth of all car miles, so a commute-only figure understates driving by roughly sixfold.
- Constrained disaggregation takes a quantity that is measured reliably, the average miles driven per person across a whole class of places, and distributes it among the neighbourhoods in that class using local signals, in a way that preserves the class average. The class totals are held fixed; only their distribution between neighbourhoods is estimated.
- The anchor is NTS9904: car-driver miles per person by 2021 rural-urban class of residence, about 2,500 miles per person in dense cities rising to roughly 5,200 in the countryside. Because it is measured by where people live, it carries the urban-to-rural driving gradient without through-traffic.
- Within each class, a neighbourhood's share is raised or lowered by its car ownership (cars per person, Census TS045) and, more gently, its commute distance (Census TS058), so lower-ownership, shorter-commute places receive fewer miles.
- The population-weighted average of the distributed miles in each class is constrained to the survey figure, so the class marginal is reproduced exactly.
- Energy is miles times household size times the local fleet's energy per mile, where a petrol car uses about 0.93 kWh per mile and an electric one about 0.32, blended by the area's share of electric vehicles (DVLA).
- One assumption is free: how strongly commute distance pulls the estimate, set by an elasticity of 0.30. The analysis reports how little the result moves when it is varied.

*Reproduce: `stats/travel_energy.py`.*

| car travel, kWh ph/yr | Flat | Terraced | Semi | Detached | ratio ph | ratio pp |
| --- | --: | --: | --: | --: | --: | --: |
| NTS-anchored | 3,240 | 5,088 | 6,660 | 9,272 | 3.23× | 2.47× |

Car travel accounts for 24–37% of all household energy. Combining the two:

| total energy, kWh ph/yr | Flat | Terraced | Semi | Detached | ratio ph | ratio pp |
| --- | --: | --: | --: | --: | --: | --: |
| heat + car travel | 13,674 | 18,265 | 20,564 | 23,832 | **2.02×** | **1.54×** |

In each table the dwelling-type columns are observed medians; the ratio is the compositional flat-to-detached estimate (ph = per household, pp = per person), so it is not the quotient of the columns.

![Household energy by dwelling type (compositional pure-type predictions): heat plus car travel rises flat to detached, a 2.0× gap.](../stats/figures/argument/energy_gradient.png)

## Access

Access is the count of things reachable from a neighbourhood, measured as network distance along the road (OS Open Roads, via cityseer) rather than straight-line. Because it is a property of the location, it is the same however the household is counted: per home or per person makes no difference to what is within reach. Three kinds of thing are counted, each in its own unit: amenities (everyday destinations: GPs, pharmacies, hospitals, schools, food outlets, supermarkets, greenspace), jobs (the total number of jobs reachable, summing the job count at each workplace), and people (the total resident population reachable). Each is read at three points on one ruler: a short walk (1.6 km), the area's own car catchment (how far its residents typically drive), and a long drive (25.6 km). What is reachable on foot is a subset of the catchment, which is a subset of the long drive.

How access is measured:

- Every count is measured as network distance over Ordnance Survey Open Roads, using the cityseer routing engine. The England street network is built once (about 3.6 million junctions), and reach is measured outward from each neighbourhood along it.
- From each Output Area, the reachable count is read at every step from a short walk (1,600 m) out to a long drive (25,600 m). The on-foot figure is a subset of the drivable one, the same ruler read closer in.
- Amenities are a count of seven everyday destinations (GPs, pharmacies and hospitals from the NHS; schools from GIAS; food outlets and supermarkets from the FSA; greenspace from Ordnance Survey). Jobs are the total jobs reachable: each workplace contributes the number of jobs it holds (Census WP101EW), so a large employer counts for more than a small one, rather than each workplace counting as one. People are likewise the total residents reachable.
- The access ratios come from the same compositional model, holding income equal but not density. Density is the mechanism by which compact form delivers access, so controlling for it would remove the effect under study. Access counts are non-negative with frequent zeros (many detached areas have no GP within a walk), so the model uses a Poisson count form, whose fitted values are constrained to be positive.

| within reach (median) | Flat | Terraced | Semi | Detached | flat:det |
| --- | --: | --: | --: | --: | --: |
| amenities, on foot | 209 | 119 | 67 | 22 | **23.9×** |
| amenities, own catchment | 2,531 | 2,255 | 2,765 | 2,776 | 1.2× |
| amenities, 25 km | 20,812 | 9,950 | 8,796 | 4,653 | 10.4× |
| jobs, on foot | 6,927 | 3,790 | 2,100 | 598 | **52.4×** |
| jobs, own catchment | 102,652 | 87,077 | 107,065 | 101,215 | 1.7× |
| jobs, 25 km | 807,658 | 382,638 | 337,938 | 173,447 | 14.3× |
| people, on foot | 17,838 | 11,861 | 8,207 | 2,766 | **12.5×** |
| people, own catchment | 255,216 | 236,228 | 285,772 | 270,115 | 1.2× |
| people, 25 km | 2,343,165 | 1,032,734 | 913,638 | 472,236 | 11.1× |

The dwelling-type columns are observed medians; the flat:det ratio is the compositional estimate. The own-catchment row reads each area at its own typical car-trip distance: a detached area reaches a similar raw count there, because it drives much further to do so, so the on-foot gap nearly closes on count and the rate below prices that extra driving in energy. For context, a flat neighbourhood holds about 79 people per hectare against a detached one's 14, a factor of 5.7.

*Reproduce: `stats/access_profile.py` (network access and the rate).*

![Amenities reachable against network distance by dwelling type (compositional pure-type predictions): a flat reaches about 24× a detached on foot, about 10× at a 25 km drive.](../stats/figures/argument/access_curve.png)

## The rate

For access the flat is ahead at every distance, and the gap narrows as distance grows. On foot a flat reaches roughly 24 times the amenities, 52 times the jobs and 12 times the people of a detached neighbourhood. At a 25 km drive, where a detached home can reach into denser places, the flat is still 10 to 14 times ahead. For energy the direction reverses: a detached home spends about 1.4 times the heat, 3.2 times the car energy, and 2.0 times the total per household. The rate combines the two: a flat returns about 6.3 times the access per kilowatt-hour it spends.

Why it differs from the on-foot gap:

- Each area's own catchment is the distance its residents typically drive: its NTS mileage divided by about 370 car trips per person per year, a trip distance capped between a short walk (1.6 km) and 25.6 km. Access per kilowatt-hour is the amenities reachable within that catchment divided by the area's car-travel energy, compared pure-flat against pure-detached as for every other ratio, giving 6.3×. It is the everyday reach each kilowatt-hour of driving buys.
- The on-foot gap (about 24×) holds distance fixed: at the same reach the flat has far more around it. A detached home reaches its own larger catchment only by spending energy to compensate, raising its count until the two nearly converge (about 1.2× apart). It spends substantially more fuel to do so, so per kilowatt-hour the flat is still 6.3× ahead.

![Amenities reachable per kWh of car travel by dwelling type (compositional pure-type predictions): a flat returns about 6.3× a detached home.](../stats/figures/argument/access_per_kwh.png)

## Lock-in

To test whether decarbonisation closes the gap, the energy is recomputed with each home at best-practice insulation and a fully electric fleet.

How the optimised scenario is computed:

- For best-practice fabric, each area's metered gas (space and water heating) is scaled by the EPC fabric-improvement ratio, potential intensity over current intensity, both EPC-modelled so the performance gap cancels (the median improvement is about half). Metered electricity (appliances, lighting) is left unchanged, since insulation does not affect it. Anchoring to the metered bill keeps the scenario on the same scale as the headline figures, and a best-insulated detached home is still a larger home that loses more heat.
- The ratio is each area's own, so it reflects local headroom rather than a blanket cut. The EPC's potential rating is its assessor's estimate of what the dwelling could reach after cost-effective measures, so an area whose homes are already efficient has a potential close to its current rating and barely changes, while a poorly rated area is cut much further. It is the median over the EPCs present in the area (about two-thirds of households hold one), and the headroom itself differs by type: flats are already efficient and gain least (a cut under a third), while terraced, semi and detached can roughly halve.
- For full electrification, car energy is recomputed at the electric fleet's energy per mile with the miles unchanged: technology lowers the energy per mile, not the distance the form forces.
- Access is unchanged by construction. No fabric or drivetrain change brings a school or shop closer, so the access axis is identical before and after.

| total energy gap (compositional) | per household | per person |
| --- | --: | --: |
| now | 2.02× | 1.54× |
| after best insulation + a full electric fleet | 1.50× | 1.15× |

The energy gap closes only part way: per household from 2.02 to 1.50 times, per person from 1.54 to 1.15 times. About half the excess survives, and a real residual remains even per resident. It splits across both halves of the form: a best-insulated detached home is still bigger, so it still loses more heat; and electrification lowers the energy per mile but not the miles, so a detached home still drives substantially further. Technology improves the efficiency of each unit but leaves the structural quantities, floor area and distance, unchanged.

The access gap does not move, because neither insulation nor electrification brings a school, a job or a shop closer to a house built far from them. The inefficiency of dispersed form is not removed by technology; it is fixed in the street layout, which changes only when places are rebuilt, over generations rather than product cycles. Access therefore has to be measured and planned for directly.

*Reproduce: `stats/lock_in.py`.*

## The NEPI scorecard, Atlas and models

The measure will be provided as three things: a NEPI scorecard, an EPC-style rating for neighbourhoods rather than buildings; an Atlas to explore the ratings; and XGBoost models that predict a neighbourhood's NEPI from its form, fabric and fleet, so different combinations of those inputs can be simulated. The models also carry a set of pre-defined scenarios, such as full electrification of the vehicle fleet or buildings brought to best-practice thermal efficiency, applied to a neighbourhood's inputs and re-scored so its NEPI under each can be read off.

*Status: these three are planned outputs, not yet built. The measured findings above stand on their own; the scorecard, Atlas and models are the intended means of delivering them.*
