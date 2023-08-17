#include "HittableList.h"

HittableList* HittableList::instance = nullptr;

HittableList& HittableList::Get()
{
    if (!instance)
    {
        instance = new HittableList();
        instance->objects = vector<shared_ptr<Hittable>>();
    }
    return *instance;
}

void HittableList::Clear()
{
    objects.clear();
}

void HittableList::Add(shared_ptr<Hittable> object)
{
    objects.push_back(object);
}

bool HittableList::Hit(const Ray& ray, Interval rayT, HitRecord& rec) const
{
    HitRecord tempRec;
    bool hitAnything = false;
    double closest = rayT.Max;
    
    for (const auto& object : objects)
    {
        if (object.get()->Hit(ray, Interval(rayT.Min, closest), tempRec))
        {
            hitAnything = true;
            closest = tempRec.T;
            rec = tempRec;
        }
    }

    return hitAnything;
}
