import hashlib
import json
import datetime
import os

class Block:
    """Represents a single block in the blockchain"""
    
    def __init__(self, index, timestamp, data, previous_hash):
        self.index = index
        self.timestamp = timestamp
        self.data = data
        self.previous_hash = previous_hash
        self.nonce = 0
        self.hash = self.calculate_hash()
    
    def calculate_hash(self):
        """Calculate SHA-256 hash of the block"""
        block_string = json.dumps({
            "index": self.index,
            "timestamp": str(self.timestamp),
            "data": self.data,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def mine_block(self, difficulty=2):
        """Mine the block with proof of work"""
        target = "0" * difficulty
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.calculate_hash()
        print(f"✅ Block mined: {self.hash}")
    
    def to_dict(self):
        """Convert block to dictionary"""
        return {
            "index": self.index,
            "timestamp": str(self.timestamp),
            "data": self.data,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
            "hash": self.hash
        }


class Blockchain:
    """Blockchain for storing crop disease predictions"""
    
    def __init__(self, storage_file='blockchain_data.json'):
        self.chain = []
        self.storage_file = storage_file
        self.difficulty = 2  # Mining difficulty
        
        # Load existing blockchain or create genesis block
        if os.path.exists(self.storage_file):
            self.load_chain()
        else:
            self.create_genesis_block()
    
    def create_genesis_block(self):
        """Create the first block in the chain"""
        genesis_block = Block(0, datetime.datetime.now(), {
            "message": "Genesis Block - Plant Disease Detection System",
            "system": "AI Crop Monitoring with Blockchain"
        }, "0")
        genesis_block.mine_block(self.difficulty)
        self.chain.append(genesis_block)
        self.save_chain()
        print("🔗 Genesis block created")
    
    def get_latest_block(self):
        """Get the most recent block"""
        return self.chain[-1]
    
    def add_prediction_block(self, prediction_data):
        """Add a new prediction to the blockchain"""
        latest_block = self.get_latest_block()
        new_index = latest_block.index + 1
        
        # Create block with prediction data
        block_data = {
            "prediction_id": f"PRED_{new_index}",
            "timestamp": str(datetime.datetime.now()),
            "disease": prediction_data.get('disease'),
            "moisture": prediction_data.get('moisture'),
            "confidence": prediction_data.get('confidence'),
            "image_name": prediction_data.get('image_name', 'unknown'),
            "verified": True
        }
        
        new_block = Block(
            index=new_index,
            timestamp=datetime.datetime.now(),
            data=block_data,
            previous_hash=latest_block.hash
        )
        
        # Mine the block
        print(f"⛏️  Mining block {new_index}...")
        new_block.mine_block(self.difficulty)
        
        self.chain.append(new_block)
        self.save_chain()
        
        print(f"✅ Block {new_index} added to blockchain")
        return new_block
    
    def is_chain_valid(self):
        """Verify the integrity of the blockchain"""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]
            
            # Check if current block's hash is correct
            if current_block.hash != current_block.calculate_hash():
                print(f"❌ Block {i} has been tampered with!")
                return False
            
            # Check if previous hash matches
            if current_block.previous_hash != previous_block.hash:
                print(f"❌ Block {i} has invalid previous hash!")
                return False
        
        return True
    
    def save_chain(self):
        """Save blockchain to file"""
        chain_data = [block.to_dict() for block in self.chain]
        with open(self.storage_file, 'w') as f:
            json.dump(chain_data, f, indent=4)
    
    def load_chain(self):
        """Load blockchain from file"""
        try:
            with open(self.storage_file, 'r') as f:
                chain_data = json.load(f)
            
            self.chain = []
            for block_dict in chain_data:
                block = Block(
                    index=block_dict['index'],
                    timestamp=datetime.datetime.fromisoformat(block_dict['timestamp']),
                    data=block_dict['data'],
                    previous_hash=block_dict['previous_hash']
                )
                block.nonce = block_dict['nonce']
                block.hash = block_dict['hash']
                self.chain.append(block)
            
            print(f"🔗 Blockchain loaded: {len(self.chain)} blocks")
        except Exception as e:
            print(f"❌ Error loading blockchain: {e}")
            self.create_genesis_block()
    
    def get_chain_data(self):
        """Get all blockchain data"""
        return [block.to_dict() for block in self.chain]
    
    def get_block_by_index(self, index):
        """Get a specific block by index"""
        if 0 <= index < len(self.chain):
            return self.chain[index].to_dict()
        return None
    
    def get_predictions_history(self):
        """Get all predictions from blockchain"""
        predictions = []
        for block in self.chain[1:]:  # Skip genesis block
            if 'disease' in block.data:
                predictions.append({
                    'block_index': block.index,
                    'prediction_id': block.data.get('prediction_id'),
                    'timestamp': block.data.get('timestamp'),
                    'disease': block.data.get('disease'),
                    'moisture': block.data.get('moisture'),
                    'confidence': block.data.get('confidence'),
                    'block_hash': block.hash
                })
        return predictions
    
    def search_by_disease(self, disease_name):
        """Search predictions by disease name"""
        results = []
        for block in self.chain[1:]:
            if block.data.get('disease', '').lower() == disease_name.lower():
                results.append(block.to_dict())
        return results


# Test the blockchain
if __name__ == "__main__":
    print("\n🔗 Testing Blockchain System")
    print("="*50)
    
    # Create blockchain
    bc = Blockchain()
    
    # Add test predictions
    test_predictions = [
        {"disease": "bacterial_blight", "moisture": 75, "confidence": 0.92, "image_name": "test1.jpg"},
        {"disease": "caterpillar", "moisture": 68, "confidence": 0.88, "image_name": "test2.jpg"},
        {"disease": "healthy", "moisture": 82, "confidence": 0.95, "image_name": "test3.jpg"}
    ]
    
    for pred in test_predictions:
        bc.add_prediction_block(pred)
    
    # Verify chain
    print("\n🔍 Verifying blockchain integrity...")
    is_valid = bc.is_chain_valid()
    print(f"Blockchain valid: {'✅ YES' if is_valid else '❌ NO'}")
    
    # Show predictions history
    print("\n📊 Predictions History:")
    history = bc.get_predictions_history()
    for pred in history:
        print(f"  • {pred['disease']} - Moisture: {pred['moisture']}% - Block: {pred['block_index']}")
    
    print("\n✅ Blockchain test complete!")