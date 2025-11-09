// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract CropMonitoring {
    struct CropData {
        string disease;
        uint256 moisture;
        uint256 timestamp;
    }

    mapping(uint256 => CropData) public cropRecords;
    uint256 public recordCount;

    event NewRecord(uint256 indexed recordId, string disease, uint256 moisture, uint256 timestamp);

    function logData(string memory _disease, uint256 _moisture) public {
        recordCount++;
        cropRecords[recordCount] = CropData(_disease, _moisture, block.timestamp);
        emit NewRecord(recordCount, _disease, _moisture, block.timestamp);
    }

    function getRecord(uint256 _recordId) public view returns (string memory, uint256, uint256) {
        CropData memory record = cropRecords[_recordId];
        return (record.disease, record.moisture, record.timestamp);
    }
}