namespace TopstepX;

internal record AuthResp(string token, bool success, int errorCode, string? errorMessage);

internal record AccountSearchResp(AccountDto[] accounts, bool success, int errorCode, string? errorMessage);
internal record AccountDto(int id, string name, bool canTrade, bool isVisible, bool simulated);

internal record ContractsResp(ContractDto[] contracts, bool success, int errorCode, string? errorMessage);
internal record ContractDto(string id, string name, string description, decimal tickSize, decimal tickValue, bool activeContract, string symbolId);
