namespace TopstepX;

public record AuthResp(string token, bool success, int errorCode, string? errorMessage);

public record AccountSearchResp(AccountDto[] accounts, bool success, int errorCode, string? errorMessage);
public record AccountDto(int id, string name, bool canTrade, bool isVisible, bool simulated);

public record ContractsResp(ContractDto[] contracts, bool success, int errorCode, string? errorMessage);
public record ContractDto(string id, string name, string description, decimal tickSize, decimal tickValue, bool activeContract, string symbolId);
