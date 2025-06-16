sim=require'sim'

function sysCall_init()
    maxScanDistance=5
    showLines=true
    generateData=true
    rangeData=false -- if false, then X/Y/Z data rel to sensor base
    discardMaxDistPts=true

    self=sim.getObject('..')
    visionSensors={sim.getObject("../sensor1"),sim.getObject("../sensor2")}
    local collection=sim.createCollection(0)
    sim.addItemToCollection(collection,sim.handle_all,-1,0)
    sim.addItemToCollection(collection,sim.handle_tree,self,1)
    sim.setObjectInt32Param(visionSensors[1],sim.visionintparam_entity_to_render,collection)
    sim.setObjectInt32Param(visionSensors[2],sim.visionintparam_entity_to_render,collection)
    sim.setObjectFloatParam(visionSensors[1],sim.visionfloatparam_far_clipping,maxScanDistance)
    sim.setObjectFloatParam(visionSensors[2],sim.visionfloatparam_far_clipping,maxScanDistance)
    red={1,0,0}
    lines=sim.addDrawingObject(sim.drawing_lines,1,0,-1,10000,red)
    
    simGridMap.init({0.1,50,0.9,0.1,5,0.1,0.1,0.8,0.7,0.3}, true)
    laserPoints = {}
    lidar_handle = sim.getObject("/youBot/fastHokuyo")
end

function sysCall_cleanup() 
    sim.removeDrawingObject(lines)
    simGridMap.release()
end 

function getLaserPoints(laserScan)
    
    local laserPts={}
    local j=1
    
    for i=1,#laserScan,3 do
        laserPts[j]={laserScan[i],laserScan[i+1]}
        j=j+1
    end
    return laserPts
end

function updateRobotPose()
    local pose
    position=sim.getObjectPosition(lidar_handle,-1)
    orientation=sim.getObjectOrientation(lidar_handle,-1)
    pose={position[1],position[2],orientation[3]}
    return pose
end

function sysCall_sensing() 
    local measuredData={}
    local lidar_pose = sim.getObjectPosition(lidar_handle,sim.handle_world)
    sim.addDrawingObjectItem(lines,nil)
    for i=1,2,1 do
        local r,t,u=sim.readVisionSensor(visionSensors[i])
        if u then
            local sensorM=sim.getObjectMatrix(visionSensors[i])
            local relRefM=sim.getObjectMatrix(self)
            relRefM=sim.getMatrixInverse(relRefM)
            relRefM=sim.multiplyMatrices(relRefM,sensorM)
            local p={0,0,0}
            p=sim.multiplyVector(sensorM,p)
            t={p[1],p[2],p[3],0,0,0}
            for j=0,u[2]-1,1 do
                for k=0,u[1]-1,1 do
                    local w=2+4*(j*u[1]+k)
                    local v={u[w+1],u[w+2],u[w+3],u[w+4]}
                    if generateData then
                        if rangeData then
                            table.insert(measuredData,v[4])
                        else
                            if v[4]<maxScanDistance*0.9999 or not discardMaxDistPts then
                                p=sim.multiplyVector(relRefM,v)
                                table.insert(measuredData,p[1])
                                table.insert(measuredData,p[2])
                                table.insert(measuredData,p[3])
                            end
                        end
                    end
                    if showLines then
                        p=sim.multiplyVector(sensorM,v)
                        t[4]=p[1]
                        t[5]=p[2]
                        t[6]=p[3]
                        sim.addDrawingObjectItem(lines,t)
                    end
                end
            end
        end
    end
    laserPoints = getLaserPoints(measuredData)
    local robo_pose = updateRobotPose()
    simGridMap.updateMapLaser(laserPoints,robo_pose)
    if #measuredData > 0 then
        --sim.setFloatProperty(sim.handle_scene ,'signal.myFloatSignal',9.01)
        sim.setBufferProperty(sim.handle_scene, "customData.myTag", sim.packTable(measuredData))

    end
end 
