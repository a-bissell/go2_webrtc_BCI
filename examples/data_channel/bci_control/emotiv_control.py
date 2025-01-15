import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from typing import Optional

from cortex import Cortex
from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.constants import RTC_TOPIC, SPORT_CMD

@dataclass
class BCIConfig:
    """Configuration for BCI control parameters"""
    client_id: str = "your_emotiv_client_id"
    client_secret: str = "your_emotiv_client_secret"
    command_threshold: float = 0.5  # Threshold for mental command detection
    profile_name: str = "Go2Control"  # Name for the training profile

@dataclass
class RobotConfig:
    """Configuration for Go2 robot parameters"""
    connection_method: WebRTCConnectionMethod = WebRTCConnectionMethod.LocalSTA
    ip: str = "192.168.8.181"
    move_speed: float = 0.5  # Movement speed (0-1)
    move_duration: float = 2.0  # Duration of movement in seconds

class BCIRobotController:
    def __init__(self, bci_config: BCIConfig, robot_config: RobotConfig):
        self.bci_config = bci_config
        self.robot_config = robot_config
        self.cortex = Cortex(self.bci_config.client_id, self.bci_config.client_secret)
        self.robot_conn: Optional[Go2WebRTCConnection] = None
        self.running = False
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    async def setup_bci(self):
        """Initialize BCI connection and training"""
        try:
            await self.cortex.open()
            await self.cortex.request_access()
            await self.cortex.authenticate()
            
            # Get headset details
            headsets = await self.cortex.query_headsets()
            if len(headsets) < 1:
                raise ValueError("No headset detected")
            
            # Connect to first available headset
            await self.cortex.connect_headset(headsets[0]['id'])
            await self.cortex.create_session(activate=True)
            
            # Load or create training profile
            profiles = await self.cortex.query_profile()
            if self.bci_config.profile_name not in profiles:
                await self.setup_training()
            else:
                await self.cortex.load_profile(self.bci_config.profile_name)
            
            self.logger.info("BCI setup completed successfully")
            
        except Exception as e:
            self.logger.error(f"BCI setup failed: {str(e)}")
            raise

    async def setup_training(self):
        """Setup and run mental command training"""
        try:
            # Create new profile
            await self.cortex.setup_profile(self.bci_config.profile_name)
            
            # Train "push" command
            self.logger.info("Starting mental command training for 'push'...")
            await self.cortex.train_mc("push")
            
            # Save the training profile
            await self.cortex.save_profile()
            self.logger.info("Training completed and profile saved")
            
        except Exception as e:
            self.logger.error(f"Training setup failed: {str(e)}")
            raise

    async def setup_robot(self):
        """Initialize robot connection"""
        try:
            self.robot_conn = Go2WebRTCConnection(
                self.robot_config.connection_method,
                ip=self.robot_config.ip
            )
            await self.robot_conn.connect()
            
            # Ensure robot is in normal mode
            await self._set_motion_mode("normal")
            self.logger.info("Robot connection established")
            
        except Exception as e:
            self.logger.error(f"Robot setup failed: {str(e)}")
            raise

    async def _set_motion_mode(self, mode: str):
        """Set robot motion mode"""
        await self.robot_conn.datachannel.pub_sub.publish_request_new(
            RTC_TOPIC["MOTION_SWITCHER"],
            {
                "api_id": 1002,
                "parameter": {"name": mode}
            }
        )
        await asyncio.sleep(2)  # Wait for mode change

    async def move_forward(self):
        """Execute forward movement"""
        try:
            await self.robot_conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"],
                {
                    "api_id": SPORT_CMD["Move"],
                    "parameter": {
                        "x": self.robot_config.move_speed,
                        "y": 0,
                        "z": 0
                    }
                }
            )
            self.logger.info("Moving forward")
            await asyncio.sleep(self.robot_config.move_duration)
            
            # Stop movement
            await self.robot_conn.datachannel.pub_sub.publish_request_new(
                RTC_TOPIC["SPORT_MOD"],
                {
                    "api_id": SPORT_CMD["Move"],
                    "parameter": {"x": 0, "y": 0, "z": 0}
                }
            )
            
        except Exception as e:
            self.logger.error(f"Movement execution failed: {str(e)}")

    async def process_mental_commands(self, mental_command_data):
        """Process incoming mental command data"""
        if not mental_command_data:
            return

        command = mental_command_data.get('command')
        power = mental_command_data.get('power', 0)

        if (command == 'push' and 
            power >= self.bci_config.command_threshold):
            self.logger.info(f"Push command detected with power: {power}")
            await self.move_forward()

    async def run(self):
        """Main control loop"""
        try:
            await self.setup_bci()
            await self.setup_robot()
            
            self.running = True
            self.logger.info("Starting mental command monitoring...")
            
            # Subscribe to mental command stream
            await self.cortex.subscribe(['com'])
            
            while self.running:
                mental_command = await self.cortex.receive_data()
                await self.process_mental_commands(mental_command)
                
        except Exception as e:
            self.logger.error(f"Runtime error: {str(e)}")
        finally:
            await self.cleanup()

    async def cleanup(self):
        """Cleanup resources"""
        self.running = False
        if self.cortex:
            await self.cortex.close()
        if self.robot_conn:
            await self.robot_conn.disconnect()
        self.logger.info("Cleanup completed")

async def main():
    # Configure your parameters
    bci_config = BCIConfig(
        client_id="your_emotiv_client_id",
        client_secret="your_emotiv_client_secret",
        command_threshold=0.5
    )
    
    robot_config = RobotConfig(
        connection_method=WebRTCConnectionMethod.LocalSTA,
        ip="192.168.8.181",
        move_speed=0.5,
        move_duration=2.0
    )
    
    controller = BCIRobotController(bci_config, robot_config)
    
    try:
        await controller.run()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    finally:
        await controller.cleanup()

if __name__ == "__main__":
    asyncio.run(main()) 