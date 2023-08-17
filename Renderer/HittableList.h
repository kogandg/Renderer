#pragma once
#include "Hittable.h"
#include "Helpers.h"

#include <memory>
#include <vector>


using namespace std;

class HittableList : public Hittable
{
private:
	HittableList() = default;
	~HittableList() = default;
	static HittableList* instance;
	vector<shared_ptr<Hittable>> objects;

public:
	static HittableList& Get();

	void Clear();
	void Add(shared_ptr<Hittable> object);
	

	HittableList(const HittableList&) = delete;
	HittableList& operator=(const HittableList&) = delete;


	// Inherited via Hittable
	bool Hit(const Ray& ray, Interval rayT, HitRecord& rec) const override;

};