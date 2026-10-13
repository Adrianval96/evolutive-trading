from .catalog import STRATEGY_PARAMETERS
from .model import Agent, AgentSpec, InvalidRunError, MarketSelection, Run, RunRequest

INTERVAL = "1d"


def normalize_agent(index: int, spec: AgentSpec) -> Agent:
    strategy_id = spec.strategy
    strategy_parameters = STRATEGY_PARAMETERS.get(strategy_id)
    if strategy_parameters is None:
        raise InvalidRunError(f'Unknown strategy "{strategy_id}".')
    label = spec.label or strategy_id
    unknown_names = sorted(set(spec.parameters) - {parameter.name for parameter in strategy_parameters})
    if unknown_names:
        raise InvalidRunError(f'{label}: Unknown parameter "{unknown_names[0]}" for {strategy_id}.')
    parameters: dict[str, float] = {}
    for parameter in strategy_parameters:
        value = spec.parameters.get(parameter.name, parameter.default)
        if parameter.type == "integer":
            if value != int(value):
                raise InvalidRunError(f"{label}: {parameter.name} must be a whole number, got {value}.")
            value = int(value)
        if not parameter.min <= value <= parameter.max:
            raise InvalidRunError(f"{label}: {parameter.name} must be between {parameter.min} and {parameter.max}, got {value}.")
        parameters[parameter.name] = value
    if strategy_id == "ma-crossover" and parameters["slow"] <= parameters["fast"]:
        raise InvalidRunError(f"{label}: slow ({parameters['slow']}) must be greater than fast ({parameters['fast']}).")
    return Agent(f"agt_{index + 1}", label, strategy_id, parameters)


def check_market(selection: MarketSelection, available: dict[str, tuple[str, str]]) -> None:
    if selection.interval != INTERVAL:
        raise InvalidRunError(f'Unknown interval "{selection.interval}".')
    if selection.start >= selection.end:
        raise InvalidRunError("start must be before end.")
    for symbol in selection.symbols:
        if symbol not in available:
            raise InvalidRunError(f"{symbol} has no {INTERVAL} data.")
        first, last = available[symbol]
        if selection.start < first or selection.end > last:
            raise InvalidRunError(f"{symbol} has {INTERVAL} data from {first} to {last} only.")


def normalize_run(request: RunRequest, available: dict[str, tuple[str, str]]) -> Run:
    agents = tuple(normalize_agent(index, spec) for index, spec in enumerate(request.agents))
    check_market(request.market, available)
    return Run(agents, request.market, request.initial_cash, request.cost_bps)
